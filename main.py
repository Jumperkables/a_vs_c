# Standard imports
import os, sys
import shutil
import argparse
from tqdm import tqdm

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
#from multimodal.models import UpDownModel
from multimodal.text import PretrainedWordEmbedding

# Local imports
import misc.myutils as myutils
import misc.dset_utils as dset_utils
from datasets import VQA, GQA, pad_question_collate

"""
Pytorch_Lightning Model handling system
"""

class LXMERT(pl.LightningModule):
    def __init__(self, args, train_dset):   # Pass ans2idx from relevant dataset object
        super().__init__()
        ans2idx = train_dset.ans2idx
        self.idx2ans = {v:k for k,v in ans2idx.items()}
        n_answers = len(ans2idx)
        fc_intermediate = ((n_answers-768)//2)+768
        self.args = args
        # LXMERT Models
        self.running_sanity_check = True
        dummy_config = lambda: None; dummy_config.hidden_size = 768 #Create a dummy config
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        for name, param in self.lxmert.named_parameters():
            param.requires_grad = True
        self.classifier_fc = LxmertVisualAnswerHead(dummy_config, n_answers)
        for name, param in self.classifier_fc.named_parameters():
            param.requires_grad = True

        if args.loss == "default":
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif args.loss in ["avsc","avsc-scaled"]:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            raise NotImplementedError(f"Loss {args.loss} not implement for {args.model} net")
        # Logging for metrics
        self.valid_acc = torchmetrics.Accuracy()
        self.valid_acc_top2 = torchmetrics.Accuracy(top_k=2)
        self.valid_acc_top3 = torchmetrics.Accuracy(top_k=3)
        self.valid_acc_top5 = torchmetrics.Accuracy(top_k=5)
        self.valid_acc_top10 = torchmetrics.Accuracy(top_k=10)
        self.train_acc = torchmetrics.Accuracy()
        self.best_acc = 0
        self.best_acc_top2 = 0
        self.best_acc_top3 = 0
        self.best_acc_top5 = 0
        self.best_acc_top10 = 0

        self.predictions = {}
        self.attentions = {}

        # RUBi things
        if args.rubi == "rubi":
            self.biased_bert = BertModel.from_pretrained("bert-base-uncased")
            for name, param in self.biased_bert.named_parameters():
                param.requires_grad = False
            self.biased_classifier_fc = nn.Sequential(
                nn.Linear(768, n_answers),
                nn.BatchNorm1d(n_answers),
                nn.GELU(),
                nn.Dropout(0.2)
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
        bsz = bboxes.shape[0]
        out = self.lxmert(question, features, bboxes, output_attentions=True)
        #['language_output', 'vision_output', 'pooled_output', 'language_attentions', 'vision_attentions', 'cross_attentions']
        #lng_out, vis_out, x_out = out['language_output'], out['vision_output'], out['pooled_output']
        x_out = out['pooled_output']
        vis_attns = torch.stack(out['vision_attentions']).mean(dim=0).mean(dim=1)
        if self.args.rubi == "rubi":
            ## Language only
            lng_out_biased = self.biased_bert(question)[1].squeeze(1)
            out_biased = self.biased_classifier_fc(lng_out_biased)
        else:
            out_biased = None
        #out = torch.cat((lng_out, vis_out, x_out), dim=1)
        out = self.classifier_fc(x_out)
        return out, out_biased, vis_attns

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(nn.ParameterList([p for n,p in self.named_parameters()]), lr=self.args.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # Prepare data
        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _, q_id_ret, _ = train_batch
        return_norm_bool = return_norm >= 0.5
        norm_answer_tens = conc_answer_tens * return_norm_bool.unsqueeze(1)
        norm_answer_tens +=  abs_answer_tens * torch.logical_not(return_norm_bool).unsqueeze(1) # The tilde (~) negates the boolean torch tensor
        # question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, ret_img_id, q_id_ret, img_dims
        out, out_biased, vis_attns = self(question, bboxes, features, image) # out_biased is from potential RUBi outputs
        if self.args.loss == "default":
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                loss = self.criterion(out, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                biased_loss = loss['biased_loss']
                loss = loss['combined_loss']+biased_loss
            else:
                loss = self.criterion(out, answer.squeeze(1))
        elif self.args.loss == "avsc":
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                loss = self.criterion(out, out_biased, norm_answer_tens, biased_loss_weighting=1.0)
                biased_loss = loss['biased_loss']
                loss = loss['combined_loss']+biased_loss
            else:
                loss = self.criterion(out, norm_answer_tens)
        elif self.args.loss == "avsc-scaled":
            norm_answer_tens = norm_answer_tens/norm_answer_tens.sum(dim=1, keepdim=True)
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                loss = self.criterion(out, out_biased, norm_answer_tens, biased_loss_weighting=1.0)
                biased_loss = loss['biased_loss']
                loss = loss['combined_loss']+biased_loss
            else:
                loss = self.criterion(out, norm_answer_tens)
        train_loss = loss
        out = F.softmax(out, dim=1)
        self.log("train_loss", loss, on_step=False, on_epoch=True)#, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc(out, answer.squeeze(1)), on_step=False, on_epoch=True)
        if self.args.rubi == "rubi":
            self.log("train_biased_loss", biased_loss)#, on_step=True)
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _, q_id_ret, img_dims = val_batch
        return_norm_bool = return_norm > 0.5
        norm_answer_tens = conc_answer_tens * return_norm_bool.unsqueeze(1)
        norm_answer_tens +=  abs_answer_tens * torch.logical_not(return_norm_bool).unsqueeze(1) # The tilde (~) negates the boolean torch tensor
        out, out_biased, vis_attns = self(question, bboxes, features, image)
        if self.args.loss == "default":
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                loss = self.criterion(out, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                biased_loss = loss['biased_loss']
                loss = loss['combined_loss']+biased_loss
            else:
                loss = self.criterion(out, answer.squeeze(1))
        elif self.args.loss == "avsc":
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                loss = self.criterion(out, out_biased, norm_answer_tens, biased_loss_weighting=1.0)
                biased_loss = loss['biased_loss']
                loss = loss['combined_loss']+biased_loss
            else:
                loss = self.criterion(out, norm_answer_tens)
        elif self.args.loss == "avsc-scaled":
            norm_answer_tens = norm_answer_tens/norm_answer_tens.sum(dim=1, keepdim=True)
            if self.args.rubi == "rubi":
                raise NotImplementedError()
                loss = self.criterion(out, out_biased, norm_answer_tens, biased_loss_weighting=1.0)
                biased_loss = loss['biased_loss']
                loss = loss['combined_loss']+biased_loss
            else:
                loss = self.criterion(out, norm_answer_tens)
        valid_loss = loss
        out = F.softmax(out, dim=1)
        self.log("valid_loss", loss, on_step=False, on_epoch=True)#, on_step=True, on_epoch=True)
        self.log("valid_acc", self.valid_acc(out, answer.squeeze(1)), on_step=False, on_epoch=True)
        self.log("valid_acc_top2", self.valid_acc_top2(out, answer.squeeze(1)), on_step=False, on_epoch=True)
        self.log("valid_acc_top3", self.valid_acc_top3(out, answer.squeeze(1)), on_step=False, on_epoch=True)
        self.log("valid_acc_top5", self.valid_acc_top5(out, answer.squeeze(1)), on_step=False, on_epoch=True)
        self.log("valid_acc_top10", self.valid_acc_top10(out, answer.squeeze(1)), on_step=False, on_epoch=True)
        out = out.argmax(dim=1)
        # Move the rescaling to the forward pass
        vis_attns = vis_attns.mean(dim=1)
        for i in range(len(q_id_ret)):
            if self.args.dataset[:3] == "GQA":
                q_idx = f"{q_id_ret[i][0]}".zfill(q_id_ret[i][1])
            else:
                q_idx = f"{q_id_ret[i][0]}"
            self.predictions[q_idx] = self.idx2ans[int(out[i])]
            self.attentions[q_idx] = [((float(bboxes[i][j][0]), float(bboxes[i][j][1]), float(bboxes[i][j][2]), float(bboxes[i][j][3])), float(vis_attns[i][j])) for j in range(len(vis_attns[i]))]
        if self.args.rubi == "rubi":
            self.log("valid_biased_loss", biased_loss)#, on_step=True)
        return valid_loss

    def validation_epoch_end(self, val_step_outputs):
        current_acc = float(self.valid_acc.compute())
        current_acc_top2 = float(self.valid_acc_top2.compute())
        current_acc_top3 = float(self.valid_acc_top3.compute())
        current_acc_top5 = float(self.valid_acc_top5.compute())
        current_acc_top10 = float(self.valid_acc_top10.compute())
        if current_acc >= self.best_acc:
            if not self.running_sanity_check:
                # Save predictions and attentions to .json file to later be handled
                metrics_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints", self.args.jobname)
                if os.path.exists(metrics_dir):
                    shutil.rmtree(metrics_dir)
                os.makedirs(metrics_dir)
                myutils.save_json(self.predictions, os.path.join(metrics_dir, "predictions.json"))
                myutils.save_json(self.attentions, os.path.join(metrics_dir, "attentions.json"))
                #if self.args.dataset[:3] == "GQA":
                #    if self.args.dataset == "GQA":
                #        val_questions = "val_balanced_questions.json"
                #    os.system(f"python gqa_eval.py --tier 'val' --checkpoint_path 'checkpoints/{args.jobname}' --score_file_name 'scores.txt' --scenes 'val_sceneGraphs.json' --questions '{val_questions}' --choices 'val_choices.json' --predictions 'predictions.json' --attentions 'attentions.json' --consistency --grounding --objectFeatures")
                #    with open(os.path.join(metrics_dir, "scores.txt")) as f:
                #        scores = f.read().replace('\n', '<br />')
                #        scores = "<p>"+scores+"</p>"
                #        self.log("scores", wandb.Html(scores))
            self.running_sanity_check = False




class LxLSTM(pl.LightningModule):
    def __init__(self, args, train_dset):   # Pass ans2idx from relevant dataset object
        super().__init__()
        ans2idx = train_dset.ans2idx
        self.idx2ans = {v:k for k,v in ans2idx.items()}
        n_answers = len(ans2idx)
        self.args = args
        # LXMERT Models
        self.high_lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        # Language/Vision LSTM
        self.lng_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.vis_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        fc_intermediate = ((n_answers-768)//2)+768
        self.high_classifier_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(8960, fc_intermediate),
            nn.BatchNorm1d(fc_intermediate),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fc_intermediate, n_answers)
        )
        #self.high_classifier_fc = nn.Sequential(
        #    nn.Linear(8960, n_answers),
        #    nn.BatchNorm1d(n_answers),
        #    nn.GELU(),
        #    nn.Dropout(0.2)
        #)
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
                nn.Linear(768, n_answers),
                nn.BatchNorm1d(n_answers),
                nn.GELU(),
                nn.Dropout(0.2)
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
        lng_out_high = lng_out_high.permute(1,0,2).reshape(bsz, -1)#.contiguous().view(bsz, -1)
        vis_out_high = vis_out_high.permute(1,0,2).reshape(bsz, -1)#.contiguous().view(bsz, -1)
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
        #raise Exception("This application of Softmax might be a bug")
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
            self.high_predictions[q_idx] = self.idx2ans[int(out_high[i])]
            self.high_attentions[q_idx] = [((float(bboxes[i][j][0]), float(bboxes[i][j][1]), float(bboxes[i][j][2]), float(bboxes[i][j][3])), float(vis_attns_high[i][j])) for j in range(len(vis_attns_high[i]))]
        if self.args.rubi == "rubi":
            self.log("valid_high_biased_loss", high_biased_loss, on_step=True)#False, on_epoch=True)
        return valid_loss

    def validation_epoch_end(self, val_step_outputs):
        current_acc = float(self.valid_acc.compute())
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
        raise NotImplementedError(f"This UpDownModel from the multimodal package appears to be deprecated")
        #self.model = UpDownModel(num_ans=n_answers, tokens=tokens)
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
            self.high_predictions[q_idx] = self.idx2ans[int(out_high[i])]
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
                if self.args.dataset[:3] == "GQA":
                    # Plot 'predictions.json' without attention
                    if self.args.dataset == "GQA":
                        val_questions = "val_balanced_questions.json"
                    elif self.args.dataset == "GQA-ABSMIXED":
                        val_questions = "absMixed_val_questions.json"
                    # Plot 'high_predictions.json' with high attentions
                    os.system(f"python gqa_eval.py --tier 'val' --checkpoint_path 'checkpoints/{args.jobname}' --score_file_name 'high_scores.txt' --scenes 'val_sceneGraphs.json' --questions '{val_questions}' --choices 'val_choices.json' --predictions 'high_predictions.json' --consistency --objectFeatures")
                    with open(os.path.join(metrics_dir, "high_scores.txt")) as f:
                        scores = f.read().replace('\n', '<br />')
                        scores = "<p>"+scores+"</p>"
                        self.log("high_scores", wandb.Html(scores))




#class Hopfield_3(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
#        super().__init__()
#        self.args = args
#        ans2idx = dataset.ans2idx
#        n_answers = len(ans2idx)
#        # LXMERT Models
#        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#        # Language/Vision LSTM
#        self.lng_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        self.vis_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        # Hopfield Nets
#        self.hopfield = hpf.Hopfield(input_size = 8960, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_high, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        fc_intermediate = ((n_answers-1024)//2)+1024
#         # High-norm / low-norm may mean high abstract/concrete. But generalised for other norms
#        raise NotImplementedError("there should be fixes here for where there used to be n_answers+1 in the linear layer, though i have now changed it")
#        self.high_classifier_fc = nn.Sequential(
#            nn.Linear(8960, n_answers),
#            nn.BatchNorm1d(n_answers),
#            nn.GELU(),
#            nn.Dropout(0.2)
#        )
#        for name, param in self.lxmert.named_parameters():
#            param.requires_grad = True
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for name, param in self.lxmert.named_parameters():
#                if not("attention" in name):
#                    param.requires_grad = False
#        elif args.unfreeze == "none":
#            for name, param in self.lxmert.named_parameters():
#                param.requires_grad = False
#        if args.loss == "default":
#            self.criterion = nn.CrossEntropyLoss(reduction='mean')
#        elif args.loss in ["avsc","avsc-scaled"]:
#            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
#        else:
#            raise NotImplementedError(f"Loss {args.loss} not implement for {args.model} net")
#        # Logging for metrics
#        self.valid_acc = torchmetrics.Accuracy()
#        self.train_acc = torchmetrics.Accuracy()
#        self.best_acc = 0
#        self.high_predictions = {}
#        self.high_attentions = {}
#
#        self.automatic_optimization = False
#
#        # RUBi things
#        if args.rubi == "rubi":
#            ################
#            raise NotImplementedError("Add rubi things in")
#            #TODO RUBI MODELS
#            ################  
#            # Overwrite criterion
#            if args.loss == "default":
#                self.criterion = myutils.RUBi_Criterion(loss_type="CrossEntropyLoss")
#            elif args.loss == "avsc":
#                self.criterion = myutils.RUBi_Criterion(loss_type="BCEWithLogitsLoss")
#            else:
#                raise NotImplementedError(f"Loss {args.loss} not implement for {args.model} net")
#
#
#    def forward(self, question, bboxes, features, image):
#        # Process language
#        bsz = question.shape[0]
#        out = self.lxmert(question, features, bboxes)     #['language_output', 'vision_output', 'pooled_output']
#        lng_out, vis_out, x_out = out['language_output'], out['vision_output'], out['pooled_output']
#        # x stands for 'cross', see naming scheme in documentation
#        # Language/Vision LSTM processing
#        _, (_, lng_out) = self.lng_lstm(lng_out)
#        _, (_, vis_out) = self.vis_lstm(vis_out)
#        breakpoint()
#        print("Was this reshape fix correct?")
#        lng_out = lng_out.permute(1,0,2).reshape(bsz,-1)#.contiguous().view(bsz, -1)
#        vis_out = vis_out.permute(1,0,2).reshape(bsz,-1)#.contiguous().view(bsz, -1)
#        # Hopfield
#        out = torch.cat((lng_out, vis_out, x_out), dim=1).unsqueeze(1)
#        out = self.hopfield(out)
#        out = out.squeeze(1)
#        out = self.classifier_fc(out)
#        return out
#
#
#    def configure_optimizers(self):
#        other_optimizer = torch.optim.Adam(nn.ParameterList([p for n,p in self.named_parameters() if "lxmert" not in n]), lr=self.args.lr)
#        lxmert_optimizer = torch.optim.Adam(nn.ParameterList([p for n,p in self.named_parameters() if "lxmert" in n]), lr=self.args.lr/5)
#        return other_optimizer, lxmert_optimizer
#
#
#    def training_step(self, train_batch, batch_idx, optimizer_idx):
#        other_optimizer, lxmert_optimizer = self.optimizers()
#        other_optimizer.zero_grad()
#        lxmert_optimizer.zero_grad()
#        # Prepare data
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = train_batch
#        out = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            train_loss = self.criterion(out, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            train_loss = torch.mean(self.criterion(out, conc_answer_tens), 1)
#        out = F.softmax(out, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.manual_backward(train_loss)
#        other_optimizer.step()
#        lxmert_optimizer.step()
#        #return train_loss
#
#
#    def validation_step(self, val_batch, batch_idx):
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = val_batch
#        out = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            valid_loss = self.criterion(out, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            valid_loss = torch.mean(self.criterion(out, conc_answer_tens), 1)
#        out = F.softmax(out, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        return valid_loss




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
    parser.add_argument("--model", type=str, default="basic", choices=["lxmert", "lx-lstm", "BUTD"  ,  "basic", "induction", "bert-lstm", "hpf-0", "hpf-1", "hpf-2", "hpf-3", "dual-lx-lstm", "dual-lxforqa"], help="Which model")

    parser.add_argument_group("LXMERT Args")
    parser.add_argument("--unfreeze", type=str, default="none", choices=["heads","all","none","qa_head"], help="What parts of LXMERT to unfreeze")

    parser.add_argument_group("Hopfield Args")
    parser.add_argument("--hopfield_beta_high", type=float, default=0.7, help="When running a high-low norm network, this is the beta scaling for the high norm hopfield net")
    parser.add_argument("--hopfield_beta_low", type=float, default=0.3, help="When running a high-low norm network, this is the beta scaling for the low norm hopfield net")

    parser.add_argument_group("Dataset arguments")
    parser.add_argument("--norm", type=str, default="conc-m", help="The norm to consider in relevant models. (conc-m == mean concreteness)")
    parser.add_argument("--norm_gt", default="answer", choices=["answer", "nsubj", "qtype", "qtype-full"], help="Where to derive the norm information of the question. 'answer'=consider the concreteness of the answer, 'nsubj'=use the concreteness of the subject of the input question")
    #### VQA-CP must have one of these 2 set to non-default values
    parser.add_argument("--topk", type=int, default=-1, help="Keep the k-top scoring answers. -1 implies ignore")
    parser.add_argument("--min_ans_occ", type=int, default=-1, help="The minimum occurence threshold for keeping an answers. -1 implies ignore")
    #parser.add_argument("--norm_ans_only", action="store_true", help="only allow questions with answers that have psycholinguistic norms")
    parser.add_argument("--norm_ans_only", choices=["simlex", "expanded"], type=str, default=None, help="only questions with answers that have psycholinguistic norms or not")
    parser.add_argument("--norm_clipping", type=float, default=0., help="The threshold to clip the norms at.")
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
    return_norm = True if args.model in ["lxmert","induction","hpf-0","hpf-1","hpf-2","hpf-3","dual-lx-lstm","dual-lxforqa"] else False
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

    pin_memory = (args.device >= 0) and (args.num_workers >= 1)
    train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers, collate_fn=pad_question_collate, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dset, batch_size=args.val_bsz, num_workers=args.num_workers, collate_fn=pad_question_collate, pin_memory=pin_memory)
    #test_loader = DataLoader(test_dset, batch_size=args.val_bsz, num_workers=args.num_workers, collate_fn=pad_question_collate, pin_memory=pin_memory)
    print(f"Total length of dataset: {len(train_dset)+len(valid_dset)}")
    total_assoc = 0
    total_ctgrcl = 0
    total_either = 0
    for batch in tqdm(train_loader, total=len(train_loader)):
        for idx in range(batch[6].shape[0]):
            assoc_flag = float(batch[6][idx].sum()) != 1.0
            ctgrcl_flag = float(batch[7][idx].sum()) != 1.0
            if assoc_flag:
                total_assoc += 1
            if ctgrcl_flag:
                total_ctgrcl += 1
            if assoc_flag or ctgrcl_flag:
                total_either += 1
    for batch in tqdm(valid_loader, total=len(valid_loader)):
        for idx in range(batch[6].shape[0]):
            assoc_flag = float(batch[6][idx].sum()) != 1.0
            ctgrcl_flag = float(batch[7][idx].sum()) != 1.0
            if assoc_flag:
                total_assoc += 1
            if ctgrcl_flag:
                total_ctgrcl += 1
            if assoc_flag or ctgrcl_flag:
                total_either += 1
    print(f"Total number of answers with assoc scores: {total_assoc}/{len(train_dset)+len(valid_dset)}")
    print(f"Total number of answers with ctgrcl scores: {total_ctgrcl}/{len(train_dset)+len(valid_dset)}")
    print(f"Total number of answers with either assoc or ctgrcl scores: {total_either}/{len(train_dset)+len(valid_dset)}")
    
    # Prepare model & pytorch_lightning system
    wandb.init(entity="jumperkables", project="a_vs_c", name=args.jobname)
    wandb_logger = pl.loggers.WandbLogger(offline=not args.wandb)#, resume="allow")
    wandb_logger.log_hyperparams(args)

    ##################################
    ##################################
    ##################################
    ## CONDITIONS FOR RUNNING COMBINATIONS OF PARAMETERS (some things will not be implemented together, hopefully these checks will catch this)
    ##################################
    # TODO ERRONEOUS UNFREEZING MUST BE ADJUSTED
    if args.model not in ["dual-lx-lstm", "dual-lxforqa", "lx-lstm", "BUTD", "lxmert"]:
        raise NotImplementedError(f"So far only dual-lx-lstm model has had the erroneous unfreezing adjusted. FIX THIS")
    # TODO NOT ALL MODELS HAVE BEEN IMPLEMENTED WITH RUBi
    if (args.rubi != "none") and (args.model not in ["dual-lx-lstm", "lx-lstm", "BUTD", "lxmert"]):
        raise NotImplementedError(f"Model {args.model} has not been updated to accomodate RUBi")
    # TODO NOT ALL METRICS HAVE BEEN UPDATED TO USE TORCHMETRICS 
    if args.model not in ["dual-lx-lstm","lx-lstm", "BUTD", "lxmert"]:
        raise NotImplementedError(f"Model {args.model} does not have metrics updated to torchmetrics with ")
    if args.model not in ["dual-lx-lstm","lx-lstm", "BUTD", "lxmert"]:
        raise NotImplementedError(f"Model {args.model} has not been upgraded to handle the question_id returning")
    # TODO Metrics plotting isnt working
    if args.rubi != "none":
        raise NotImplementedError("RUBI metrics plotting isnt currently working")
    ##################################
    ##################################
    ##################################

    if args.model == "lx-lstm":
        pl_system = LxLSTM(args, train_dset)
    elif args.model == "lxmert":
        pl_system = LXMERT(args, train_dset)
    elif args.model == "BUTD":
        pl_system = BottomUpTopDown(args, train_dset)
    elif args.model == "hpf-3":
        pl_system = Hopfield_3(args, train_dset)
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
    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="valid_acc", mode="max")
    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stopping_callback], logger=wandb_logger, gpus=gpus, max_epochs=args.epochs)
    trainer.fit(pl_system, train_loader, valid_loader)
    #trainer.test(pl_system, test_loader)
