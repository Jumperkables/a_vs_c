# Standard imports
import os, sys
import shutil
import argparse

# Complex imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import LxmertConfig, LxmertForQuestionAnswering, LxmertModel, LxmertTokenizer, BertTokenizer, BertModel, BertConfig
from transformers.models.lxmert.modeling_lxmert import LxmertVisualAnswerHead
import pytorch_lightning as pl
import torchmetrics
import wandb
from multimodal.models import UpDownModel
from multimodal.text import PretrainedWordEmbedding

# Local imports
import misc.myutils as myutils
import misc.dset_utils as dset_utils
from misc.multimodal_pip_vqa_utils import process_annotations
from datasets import VQA, GQA, pad_question_collate

"""
Pytorch_Lightning Model handling system
"""
class LxLSTM(pl.LightningModule):
    def __init__(self, args, train_dset):   # Pass ans2idx from relevant dataset object
        super().__init__()
        ans2idx = train_dset.ans2idx
        n_answers = len(ans2idx)
        self.args = args
        # LXMERT Models
        self.high_lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        # Language/Vision LSTM
        self.lng_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.vis_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        fc_intermediate = ((n_answers-8960)//2)+8960
        self.high_classifier_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(8960, fc_intermediate),
            nn.BatchNorm1d(fc_intermediate),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fc_intermediate, n_answers)
        )
        for name, param in self.high_lxmert.named_parameters():
            param.requires_grad = True
        if args.unfreeze == "all":
            pass
        elif args.unfreeze == "heads":
            for name, param in self.high_lxmert.named_parameters():
                if not("attention" in name):
                    param.requires_grad = False
        elif args.unfreeze == "none":
            for name, param in self.high_lxmert.named_parameters():
                param.requires_grad = False
        if args.loss == "default":
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif args.loss in ["avsc","avsc-scaled"]:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            raise NotImplementedError(f"Loss {args.loss} not implement for {args.model} net")
        # Logging for metrics
        self.valid_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.best_acc = 0
        self.high_predictions = {}
        self.high_attentions = {}
        # TODO DEPRECATED# Correct answer ids to work out consistency/plausability etc..
        # self.correct_answers = []
        # self.correct_answers_high = []
        # Manual optimisation to allow slower training for previous layers
        self.automatic_optimization = False

        # RUBi things
        if args.rubi == "rubi":
            self.biased_bert = BertModel.from_pretrained("bert-base-uncased")
            for name, param in self.biased_bert.named_parameters():
                #if not("attention" in name):
                param.requires_grad = False
            self.biased_classifier_fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(768, fc_intermediate),#nn.Linear(4096, fc_intermediate),
                nn.BatchNorm1d(fc_intermediate),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(fc_intermediate, n_answers)
            )
            # Overwrite criterion
            if args.loss == "default":
                self.criterion = myutils.RUBi_Criterion(loss_type="CrossEntropyLoss")
            elif args.loss == "avsc":
                self.criterion = myutils.RUBi_Criterion(loss_type="BCEWithLogitsLoss")
            else:
                raise NotImplementedError(f"Loss {args.loss} not implement for {args.model} net")


    def forward(self, question, bboxes, features, image):
        # Process language
        # NOTE According to GitHub issue: https://github.com/airsplay/lxmert/issues/26, bounding boxes are of form (x0,y0,x1,y1) for lxmert
        #['language_output', 'vision_output', 'pooled_output', 'language_attentions', 'vision_attentions', 'cross_attentions']
        bsz = bboxes.shape[0]
        out_high = self.high_lxmert(question, features, bboxes, output_attentions=True)
        #['language_output', 'vision_output', 'pooled_output', 'language_attentions', 'vision_attentions', 'cross_attentions']
        lng_out_high, vis_out_high, x_out_high = out_high['language_output'], out_high['vision_output'], out_high['pooled_output']
        vis_attns_high = torch.stack(out_high['vision_attentions']).mean(dim=0).mean(dim=1)
        if self.args.rubi == "rubi":
            ## Language only
            lng_out_biased = self.biased_bert(question)[1].squeeze(1)
            out_biased = self.biased_classifier_fc(lng_out_biased)
        else:
            out_biased = None
        # x stands for 'cross', see naming scheme in documentation
        # Language/Vision LSTM processing
        _, (_, lng_out_high) = self.lng_lstm(lng_out_high)
        _, (_, vis_out_high) = self.vis_lstm(vis_out_high)
        lng_out_high = lng_out_high.permute(1,0,2).contiguous().view(bsz, -1)
        vis_out_high = vis_out_high.permute(1,0,2).contiguous().view(bsz, -1)
        out_high = torch.cat((lng_out_high, vis_out_high, x_out_high), dim=1)
        out_high = self.high_classifier_fc(out_high)
        return out_high, out_biased, vis_attns_high

    def configure_optimizers(self):
        other_optimizer = torch.optim.Adam(nn.ParameterList([p for n,p in self.named_parameters() if "lxmert" not in n]), lr=self.args.lr)
        lxmert_optimizer = torch.optim.Adam(nn.ParameterList([p for n,p in self.named_parameters() if "lxmert" in n]), lr=self.args.lr/5)
        return other_optimizer, lxmert_optimizer

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        # Prepare data
        other_optimizer, lxmert_optimizer = self.optimizers()
        other_optimizer.zero_grad()
        lxmert_optimizer.zero_grad()
        question, answer, bboxes, features, image, return_norm, _, conc_answer_tens, _, q_id_ret, _ = train_batch
        out_high, out_biased, vis_attns_high = self(question, bboxes, features, image) # out_biased is from potential RUBi outputs
        if self.args.loss == "default":
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                #['combined_loss']
                #['main_loss']
                #['biased_loss']
                high_loss = self.criterion(out_high, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                high_loss = self.criterion(out_high, answer.squeeze(1))
        elif self.args.loss == "avsc":
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                high_loss = self.criterion(out_high, out_biased, conc_answer_tens, biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                high_loss = self.criterion(out_high, conc_answer_tens)
        elif self.args.loss == "avsc-scaled":
            conc_answer_tens = conc_answer_tens/conc_answer_tens.sum(dim=1, keepdim=True)
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                high_loss = self.criterion(out_high, out_biased, conc_answer_tens, biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                high_loss = self.criterion(out_high, conc_answer_tens)
        train_loss = high_loss
        out_high = F.softmax(out_high, dim=1)
        self.log("train_loss", high_loss, on_step=True)#False, on_epoch=True)
        self.log("train_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
        if self.args.rubi == "rubi":
            self.log("train_high_biased_loss", high_biased_loss, on_step=True)#False, on_epoch=True)
        self.manual_backward(train_loss)
        other_optimizer.step()
        lxmert_optimizer.step()
        #return train_loss

    def validation_step(self, val_batch, batch_idx):
        question, answer, bboxes, features, image, return_norm, _, conc_answer_tens, _, q_id_ret, img_dims = val_batch
        out_high, out_biased, vis_attns_high = self(question, bboxes, features, image)
        if self.args.loss == "default":
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                #['combined_loss']
                #['main_loss']
                #['biased_loss']
                high_loss = self.criterion(out_high, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                high_loss = self.criterion(out_high, answer.squeeze(1))
        elif self.args.loss == "avsc":
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                high_loss = self.criterion(out_high, out_biased, conc_answer_tens, biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                high_loss = self.criterion(out_high, conc_answer_tens)
        elif self.args.loss == "avsc-scaled":
            conc_answer_tens = conc_answer_tens/conc_answer_tens.sum(dim=1, keepdim=True)
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                high_loss = self.criterion(out_high, out_biased, conc_answer_tens, biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                high_loss = self.criterion(out_high, conc_answer_tens)
        valid_loss = high_loss
        out_high = F.softmax(out_high, dim=1)
        self.log("valid_loss", high_loss, on_step=True)#False, on_epoch=True)
        self.log("valid_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
        out_high = out_high.argmax(dim=1)
        # Move the rescaling to the forward pass
        vis_attns_high = vis_attns_high.mean(dim=1)
        for i in range(len(q_id_ret)):
            if self.args.dataset[:3] == "GQA":
                q_idx = f"{q_id_ret[i][0]}".zfill(q_id_ret[i][1])
            else:
                q_idx = f"{q_id_ret[i][0]}"
            self.high_predictions[q_idx] = self.val_dataloader.dataloader.dataset.idx2ans[int(out_high[i])]
            self.high_attentions[q_idx] = [((float(bboxes[i][j][0]), float(bboxes[i][j][1]), float(bboxes[i][j][2]), float(bboxes[i][j][3])), float(vis_attns_high[i][j])) for j in range(len(vis_attns_high[i]))]
        if self.args.rubi == "rubi":
            self.log("valid_high_biased_loss", high_biased_loss, on_step=True)#False, on_epoch=True)
        return valid_loss

    def validation_epoch_end(self, val_step_outputs):
        current_acc = (self.valid_acc.correct/self.valid_acc.total) if self.valid_acc.total != 0 else 0
        if current_acc >= self.best_acc:
            if not self.trainer.running_sanity_check:
                # Save predictions and attentions to .json file to later be handled
                metrics_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints", self.args.jobname)
                if os.path.exists(metrics_dir):
                    shutil.rmtree(metrics_dir)
                os.makedirs(metrics_dir)
                myutils.save_json(self.high_predictions, os.path.join(metrics_dir, "high_predictions.json"))
                myutils.save_json(self.high_attentions, os.path.join(metrics_dir, "high_attentions.json"))
                if self.args.dataset[:3] == "GQA":
                    # Plot 'predictions.json' without attention
                    if self.args.dataset == "GQA":
                        val_questions = "val_balanced_questions.json"
                    elif self.args.dataset == "GQA-ABSMIXED":
                        val_questions = "absMixed_val_questions.json"
                    # Plot 'high_predictions.json' with high attentions
                    os.system(f"python gqa_eval.py --tier 'val' --checkpoint_path 'checkpoints/{args.jobname}' --score_file_name 'high_scores.txt' --scenes 'val_sceneGraphs.json' --questions '{val_questions}' --choices 'val_choices.json' --predictions 'high_predictions.json' --attentions 'high_attentions.json' --consistency --grounding --objectFeatures")
                    with open(os.path.join(metrics_dir, "high_scores.txt")) as f:
                        scores = f.read().replace('\n', '<br />')
                        scores = "<p>"+scores+"</p>"
                        self.log("high_scores", wandb.Html(scores))



class BottomUpTopDown(pl.LightningModule):
    def __init__(self, args, dataset):   # Pass ans2idx from relevant dataset object
        super().__init__()
        self.args = args
        ans2idx = dataset.ans2idx
        n_answers = len(ans2idx)
        tokens = dataset.tokeniser.tokens
        self.model = UpDownModel(num_ans=n_answers, tokens=tokens)
        if args.loss == "default":
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif args.loss in ["avsc","avsc-scaled"]:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            raise NotImplementedError(f"Loss {args.loss} not implement for {args.model} net")
        # Logging for metrics
        self.valid_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.best_acc = 0
        self.high_predictions = {}
        self.high_attentions = {}

        # RUBi things
        if args.rubi == "rubi":
            ################
            raise NotImplementedError("Add rubi things in")
            breakpoint()
            #TODO RUBI MODELS
            ################  
            # Overwrite criterion
            if args.loss == "default":
                self.criterion = myutils.RUBi_Criterion(loss_type="CrossEntropyLoss")
            elif args.loss == "avsc":
                self.criterion = myutils.RUBi_Criterion(loss_type="BCEWithLogitsLoss")
            else:
                raise NotImplementedError(f"Loss {args.loss} not implement for {args.model} net")


    def forward(self, batch):
        out = self.model(batch)
        return out['logits']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(nn.ParameterList([p for n,p in self.named_parameters()]), lr=self.args.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # Prepare data
        question, answer, bboxes, features, image, return_norm, _, conc_answer_tens, _, q_id_ret, img_dims = train_batch
        out_high = self({"question_tokens":question, "features":features})
        if self.args.loss == "default":
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                #['combined_loss']
                #['main_loss']
                #['biased_loss']
                high_loss = self.criterion(out_high, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                high_loss = self.criterion(out_high, answer.squeeze(1))
        elif self.args.loss == "avsc":
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                high_loss = self.criterion(out_high, out_biased, conc_answer_tens, biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                high_loss = self.criterion(out_high, conc_answer_tens)
        elif self.args.loss == "avsc-scaled":
            conc_answer_tens = conc_answer_tens/conc_answer_tens.sum(dim=1, keepdim=True)
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                high_loss = self.criterion(out_high, out_biased, conc_answer_tens, biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                high_loss = self.criterion(out_high, conc_answer_tens)
        train_loss = high_loss
        out_high = F.softmax(out_high, dim=1)
        self.log("train_loss", high_loss, on_step=True)#False, on_epoch=True)
        self.log("train_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
        if self.args.rubi == "rubi":
            self.log("train_high_biased_loss", high_biased_loss, on_step=True)#False, on_epoch=True)
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        question, answer, bboxes, features, image, return_norm, _, conc_answer_tens, _, q_id_ret, img_dims = val_batch
        out_high = self({"question_tokens":question, "features":features})
        if self.args.loss == "default":
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                #['combined_loss']
                #['main_loss']
                #['biased_loss']
                high_loss = self.criterion(out_high, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                high_loss = self.criterion(out_high, answer.squeeze(1))
        elif self.args.loss == "avsc":
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                #high_loss = self.criterion(out_high, out_biased, conc_answer_tens, biased_loss_weighting=1.0)
                #high_biased_loss = high_loss['biased_loss']
                #high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                high_loss = self.criterion(out_high, conc_answer_tens)
        elif self.args.loss == "avsc-scaled":
            conc_answer_tens = conc_answer_tens/conc_answer_tens.sum(dim=1, keepdim=True)
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                #high_loss = self.criterion(out_high, out_biased, conc_answer_tens, biased_loss_weighting=1.0)
                #high_biased_loss = high_loss['biased_loss']
                #high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                high_loss = self.criterion(out_high, conc_answer_tens)
        valid_loss = high_loss
        out_high = F.softmax(out_high, dim=1)
        self.log("valid_loss", high_loss, on_step=True)#False, on_epoch=True)
        self.log("valid_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
        out_high = out_high.argmax(dim=1)
        # Move the rescaling to the forward pass
        for i in range(len(q_id_ret)):
            if self.args.dataset[:3] == "GQA":
                q_idx = f"{q_id_ret[i][0]}".zfill(q_id_ret[i][1])
            else:
                q_idx = f"{q_id_ret[i][0]}"
            self.high_predictions[q_idx] = self.val_dataloader.dataloader.dataset.idx2ans[int(out_high[i])]
        if self.args.rubi == "rubi":
            self.log("valid_high_biased_loss", high_biased_loss, on_step=True)#False, on_epoch=True)
        return valid_loss

    def validation_epoch_end(self, val_step_outputs):
        current_acc = (self.valid_acc.correct/self.valid_acc.total) if self.valid_acc.total != 0 else 0
        if current_acc >= self.best_acc:
            if not self.trainer.running_sanity_check:
                # Save predictions and attentions to .json file to later be handled
                metrics_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints", self.args.jobname)
                if os.path.exists(metrics_dir):
                    shutil.rmtree(metrics_dir)
                os.makedirs(metrics_dir)
                myutils.save_json(self.high_predictions, os.path.join(metrics_dir, "high_predictions.json"))
                myutils.save_json(self.high_attentions, os.path.join(metrics_dir, "high_attentions.json"))
                if self.args.dataset[:3] == "GQA":
                    # Plot 'predictions.json' without attention
                    if self.args.dataset == "GQA":
                        val_questions = "val_balanced_questions.json"
                    elif self.args.dataset == "GQA-ABSMIXED":
                        val_questions = "absMixed_val_questions.json"
                    # Plot 'high_predictions.json' with high attentions
                    os.system(f"python gqa_eval.py --tier 'val' --checkpoint_path 'checkpoints/{args.jobname}' --score_file_name 'high_scores.txt' --scenes 'val_sceneGraphs.json' --questions '{val_questions}' --choices 'val_choices.json' --predictions 'high_predictions.json' --attentions 'high_attentions.json' --consistency --grounding --objectFeatures")
                    with open(os.path.join(metrics_dir, "high_scores.txt")) as f:
                        scores = f.read().replace('\n', '<br />')
                        scores = "<p>"+scores+"</p>"
                        self.log("high_scores", wandb.Html(scores))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Basic ML Arguments")
    parser.add_argument("--jobname", type=str, required=True, help="Jobname")
    parser.add_argument("--dataset", type=str.upper, required=True, choices=["VQACP","VQACP2","GQA","GQA-ABSMIXED","VQA","VQA2"], help="Choose VQA dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--bsz", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_bsz", type=int, default=100, help="Validation batch size")
    parser.add_argument("--device", type=int, default=-1, help="Which device to run things on. (-1 = CPU)")
    parser.add_argument("--wandb", action="store_true", help="Plot wandb results online")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of pytroch workers. More should increase disk reads, but will increase RAM usage. 0 = main process")

    parser.add_argument_group("Model Arguments")
    parser.add_argument("--model", type=str, default="basic", choices=["lx-lstm", "BUTD"  ,  "basic", "induction", "bert-lstm", "hpf-0", "hpf-1", "hpf-2", "hpf-3", "dual-lx-lstm", "dual-lxforqa"], help="Which model")

    parser.add_argument_group("LXMERT Args")
    parser.add_argument("--unfreeze", type=str, required=True, choices=["heads","all","none","qa_head"], help="What parts of LXMERT to unfreeze")

    parser.add_argument_group("Hopfield Args")
    parser.add_argument("--hopfield_beta_high", type=float, default=0.7, help="When running a high-low norm network, this is the beta scaling for the high norm hopfield net")
    parser.add_argument("--hopfield_beta_low", type=float, default=0.3, help="When running a high-low norm network, this is the beta scaling for the low norm hopfield net")

    parser.add_argument_group("Dataset arguments")
    parser.add_argument("--norm", type=str, default="conc-m", help="The norm to consider in relevant models. (conc-m == mean concreteness)")
    parser.add_argument("--norm_gt", default="answer", choices=["answer", "nsubj", "qtype", "qtype-full"], help="Where to derive the norm information of the question. 'answer'=consider the concreteness of the answer, 'nsubj'=use the concreteness of the subject of the input question")
    #### VQA-CP must have one of these 2 set to non-default values
    parser.add_argument("--topk", type=int, default=-1, help="Keep the k-top scoring answers. -1 implies ignore")
    parser.add_argument("--min_ans_occ", type=int, default=-1, help="The minimum occurence threshold for keeping an answers. -1 implies ignore")
    parser.add_argument("--norm_ans_only", action="store_true", help="only allow questions with answers that have psycholinguistic norms")
    parser.add_argument("--loss", type=str, default="default", choices=["default","avsc","avsc-scaled"], help="Whether or not to use a special loss")
    parser.add_argument("--rubi", type=str, default=None, choices=["none", "rubi"], help="Using the Reducing Unimodal Bias")
    parser.add_argument("--dual_loss_style", type=str, default="linear", choices=["linear", "quadr", "cubic", "4th"], help="For dual models, e.g: linear=(k/1-k), quadr=**2, cubic=**3 etc...")
    """
    --loss:
        default: Regular softmax loss across answer
        avsc:    Add to the cross entropy other answers scaled by their occurence as word-pair norms in the actual answers 
        avsc-scaled:    As above, by normalise (by division of sum) the tensor values to sum to 1
    """
    args = parser.parse_args()
    myutils.print_args(args)

    # Prepare dataloaders
    vqa_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa")

    # Checks on if you can run
    if args.dataset in ["VQA","VQA2","VQACP","VQACP2"]:
        assert (args.topk != -1) or (args.min_ans_occ != -1), f"For VQA-CP v1/2, you must set one of topk or min_ans_occ to not default. This decides which scheme to follow to keep which answers"
        assert not((args.topk != -1) and (args.min_ans_occ != -1)), f"You must leave one of topk, or min_ans_occ at default value"
    #TODO: Automate/instruct how to use the multimodal package to neatly download files
    if (args.dataset in ["VQA","VQA2","VQACP","VQACP2"]) and (not os.path.exists(os.path.join(vqa_path, "datasets"))):
        dset_utils.download_vqa()

    # Set the correct flags for dataset processing based on which model
    # TODO Consider a more elegant way to handle these flags
    objects_flag = True 
    images_flag = False
    resnet_flag = True if args.model in ["hpf-2"] else False
    return_norm = True if args.model in ["induction","hpf-0","hpf-1","hpf-2","hpf-3","dual-lx-lstm","dual-lxforqa"] else False
    return_avsc = True if args.loss in ["avsc","avsc-scaled"] else False

    if args.dataset == "VQACP":
        train_dset = VQA(args, version="cp-v1", split="train", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = VQA(args, version="cp-v1", split="test", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
    elif args.dataset == "VQACP2":
        train_dset = VQA(args, version="cp-v2", split="train", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = VQA(args, version="cp-v2", split="test", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
    elif args.dataset == "VQA":
        train_dset = VQA(args, version="v1", split="train", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = VQA(args, version="v1", split="valid", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
    elif args.dataset == "VQA2":
        train_dset = VQA(args, version="v2", split="train", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = VQA(args, version="v2", split="valid", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
    elif args.dataset == "GQA":
        # TODO Instructions for downloading GQA
        train_dset = GQA(args, split="train", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = GQA(args, split="valid", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
    elif args.dataset == "GQA-ABSMIXED":
        train_dset = GQA(args, split="train-absMixed", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
        valid_dset = GQA(args, split="valid-absMixed", objects=objects_flag, images=images_flag, resnet=resnet_flag, return_norm=return_norm, return_avsc=return_avsc)
    else:
        raise NotImplementedError(f"{args.dataset} not recognised.")

    train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers, collate_fn=pad_question_collate)
    valid_loader = DataLoader(valid_dset, batch_size=args.val_bsz, num_workers=args.num_workers, collate_fn=pad_question_collate)
    
    # Prepare model & pytorch_lightning system
    wandb_logger = pl.loggers.WandbLogger(project="a_vs_c", name=args.jobname, offline=not args.wandb)#, resume="allow")
    wandb_logger.log_hyperparams(args)

    ##################################
    ##################################
    ##################################
    ## CONDITIONS FOR RUNNING COMBINATIONS OF PARAMETERS (some things will not be implemented together, hopefully these checks will catch this)
    ##################################
    # TODO ERRONEOUS UNFREEZING MUST BE ADJUSTED
    if args.model not in ["dual-lx-lstm", "dual-lxforqa", "lx-lstm", "BUTD"]:
        raise NotImplementedError(f"So far only dual-lx-lstm model has had the erroneous unfreezing adjusted. FIX THIS")
    # TODO NOT ALL MODELS HAVE BEEN IMPLEMENTED WITH RUBi
    if (args.rubi != "none") and (args.model not in ["dual-lx-lstm", "lx-lstm", "BUTD"]):
        raise NotImplementedError(f"Model {args.model} has not been updated to accomodate RUBi")
    # TODO NOT ALL METRICS HAVE BEEN UPDATED TO USE TORCHMETRICS 
    if args.model not in ["dual-lx-lstm","lx-lstm", "BUTD"]:
        raise NotImplementedError(f"Model {args.model} does not have metrics updated to torchmetrics with ")
    if args.model not in ["dual-lx-lstm","lx-lstm", "BUTD"]:
        raise NotImplementedError(f"Model {args.model} has not been upgraded to handle the question_id returning")
    # TODO Metrics plotting isnt working
    if args.rubi != "none":
        raise NotImplementedError("RUBI metrics plotting isnt currently working")
    ##################################
    ##################################
    ##################################

    if args.model == "lx-lstm":
        pl_system = LxLSTM(args, train_dset)
    elif args.model == "BUTD":
        pl_system = BottomUpTopDown(args, train_dset)
    elif args.model == "dual-lx-lstm":
        pl_system = None#Dual_LxLSTM(args, n_answers, train_dset.ans2idx)    # Pass the ans2idx to get answer norms 
    elif args.model == "dual-lxforqa":
        pl_system = None#Dual_LxForQA(args, n_answers, train_dset.ans2idx)    # Pass the ans2idx to get answer norms 
    else:
        raise NotImplementedError("Model: {args.model} not implemented")
    if args.device == -1:
        gpus = None
    else:
        gpus = [args.device]    # TODO Implement multigpu support

    # Checkpointing and running
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='valid_acc',
        dirpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints"),
        filename=f"{args.jobname}"+'-{epoch:02d}-{valid_acc:.2f}',
        save_top_k=1,
        mode='max',
    )
    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=wandb_logger, gpus=gpus, max_epochs=args.epochs)
    trainer.fit(pl_system, train_loader, valid_loader)
