__author__ = "Jumperkables"
import os, sys
import torch
from torch import nn
import torch.nn.functional as F

# For BERT
from transformers import *

class ABC(nn.Module):
    def __init__(self, opt):
        super(ABC, self).__init__()
        self.vid_flag = "imagenet" in opt.input_streams
        self.sub_flag = "sub" in opt.input_streams
        self.vcpt_flag = "vcpt" in opt.input_streams
        self.reg_flag = "regional" in opt.input_streams
        self.regtopk_flag = (-1 != opt.regional_topk)
        self.topk = opt.topk
        self.opt = opt
        hidden_size_1 = opt.hsz1
        hidden_size_2 = opt.hsz2
        n_layers_cls = opt.n_layers_cls
        vid_feat_size = opt.vid_feat_size
        embedding_size = opt.embedding_size

        # For BERT
        vocab_size = opt.vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bert_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 300),
            nn.Tanh(),
        )

        if opt.bert in ["default"]:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        elif opt.bert == "multi_choice":
            self.bert = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        elif opt.bert == "qa":
            self.bert = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        self.bidaf = BidafAttn(hidden_size_1 * 3, method="dot")  # no parameter for dot

        self.lstm_raw = RNNEncoder(300, hidden_size_1, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm") #changed


        if self.vid_flag:
            print("activate video stream")
            if self.opt.bert is None:
                self.video_fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(2048, embedding_size),
                    nn.Tanh(),
                )
            else:
                self.video_fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(2048, 768),
                    nn.Tanh(),
                )
            self.lstm_mature_vid = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_vid = MLP(hidden_size_2*2, 1, 500, n_layers_cls)

        if self.sub_flag:
            print("activate sub stream")
            self.lstm_mature_sub = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_sub = MLP(hidden_size_2*2, 1, 500, n_layers_cls)

        if self.vcpt_flag:
            print("activate vcpt stream")
            self.lstm_mature_vcpt = RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
                                               dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_vcpt = MLP(hidden_size_2*2, 1, 500, n_layers_cls)

        if self.reg_flag and not self.regtopk_flag:
            print("activate regional stream")
            self.regional_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, embedding_size),
                nn.Tanh(),
            )
            self.lstm_mature_reg= RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_reg = MLP(hidden_size_2*2, 1, 500, n_layers_cls)

        if self.regtopk_flag:
            print("activate regional-topk stream")
            if self.opt.bert is None:
                self.regionaltopk_fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(2048, embedding_size),
                    nn.Tanh(),
                )
            else:
                self.regionaltopk_fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(2048, 768),
                    nn.Tanh(),
                )
            self.lstm_mature_regtopk= RNNEncoder(hidden_size_1 * 2 * 5, hidden_size_2, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_regtopk = MLP(hidden_size_2*2, 1, 500, n_layers_cls)



    def load_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def forward(self, q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, a4, a4_l,
                sub, sub_l, vcpt, vcpt_l, vid, vid_l, reg, reg_l, regtopk, regtopk_l):
        #import ipdb; ipdb.set_trace()
        if self.opt.bert is None:# For BERT
            e_q = self.embedding(q)
            e_a0 = self.embedding(a0)
            e_a1 = self.embedding(a1)
            e_a2 = self.embedding(a2)
            e_a3 = self.embedding(a3)
            e_a4 = self.embedding(a4)

            raw_out_q, _ = self.lstm_raw(e_q, q_l)
            raw_out_a0, _ = self.lstm_raw(e_a0, a0_l)
            raw_out_a1, _ = self.lstm_raw(e_a1, a1_l)
            raw_out_a2, _ = self.lstm_raw(e_a2, a2_l)
            raw_out_a3, _ = self.lstm_raw(e_a3, a3_l)
            raw_out_a4, _ = self.lstm_raw(e_a4, a4_l)
        else:
            e_q  = self.bert(q)[0]
            e_a0 = self.bert(a0)[0]
            e_a1 = self.bert(a1)[0]
            e_a2 = self.bert(a2)[0]
            e_a3 = self.bert(a3)[0]
            e_a4 = self.bert(a4)[0]

            raw_out_q, _ = self.lstm_raw(e_q, q_l)
            raw_out_a0, _ = self.lstm_raw(e_a0, a0_l)
            raw_out_a1, _ = self.lstm_raw(e_a1, a1_l)
            raw_out_a2, _ = self.lstm_raw(e_a2, a2_l)
            raw_out_a3, _ = self.lstm_raw(e_a3, a3_l)
            raw_out_a4, _ = self.lstm_raw(e_a4, a4_l)

        if self.sub_flag and ('sub' not in self.opt.disable_streams):
            if(self.opt.bert is None):# For BERT
                e_sub  = self.embedding(sub) #Subtitles embedded
            else:
                e_sub  = self.bert(sub)[0]
            raw_out_sub, _ = self.lstm_raw(e_sub, sub_l) #through lstm
            sub_out = self.stream_processor(self.lstm_mature_sub, self.classifier_sub, raw_out_sub, sub_l,
                                            raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l,
                                            raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l) #Fusion happens in here for subtitles
        else:
            sub_out = 0

        if self.vcpt_flag and ('vcpt' not in self.opt.disable_streams):
            
            vcpt = self.bert_fc(vcpt)
            e_vcpt = self.embedding(vcpt)

            raw_out_vcpt, _ = self.lstm_raw(e_vcpt, vcpt_l)
            vcpt_out = self.stream_processor(self.lstm_mature_vcpt, self.classifier_vcpt, raw_out_vcpt, vcpt_l,
                                             raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l,
                                             raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)
        else:
            vcpt_out = 0

        if self.vid_flag and ('vid' not in self.opt.disable_streams):
            e_vid = self.video_fc(vid)
            
            raw_out_vid, _ = self.lstm_raw(e_vid, vid_l)
            vid_out = self.stream_processor(self.lstm_mature_vid, self.classifier_vid, raw_out_vid, vid_l,
                                            raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l,
                                            raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)
        else:
            vid_out = 0

        #### Reg ####
        if self.reg_flag and not self.regtopk_flag:
            e_reg = self.regional_fc(reg)
            raw_out_reg, _ = self.lstm_raw(e_reg, reg_l)
            reg_out = self.stream_processor(self.lstm_mature_reg, self.classifier_reg, raw_out_reg, reg_l,
                                            raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l,
                                            raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)
        else:
            reg_out = 0

        #### Reg Top k ####
        if self.regtopk_flag and ('regional' not in self.opt.disable_streams):
            e_regtopk = self.regionaltopk_fc(regtopk)
            raw_out_regtopk, _ = self.lstm_raw(e_regtopk, regtopk_l)
            regtopk_out = self.stream_processor(self.lstm_mature_regtopk, self.classifier_regtopk, raw_out_regtopk, regtopk_l,
                                            raw_out_q, q_l, raw_out_a0, a0_l, raw_out_a1, a1_l,
                                            raw_out_a2, a2_l, raw_out_a3, a3_l, raw_out_a4, a4_l)
        else:
            regtopk_out = 0
        out = sub_out + vcpt_out + vid_out + reg_out + regtopk_out # adding zeros has no effect on backward
        if self.opt.lanecheck:
            return sub_out, vcpt_out, vid_out, reg_out, regtopk_out, out.squeeze()
        else:
            return out.squeeze()

    def stream_processor(self, lstm_mature, classifier, ctx_embed, ctx_l,
                         q_embed, q_l, a0_embed, a0_l, a1_embed, a1_l, a2_embed, a2_l, a3_embed, a3_l, a4_embed, a4_l):
        u_q, _ = self.bidaf(ctx_embed, ctx_l, q_embed, q_l)
        u_a0, _ = self.bidaf(ctx_embed, ctx_l, a0_embed, a0_l)
        u_a1, _ = self.bidaf(ctx_embed, ctx_l, a1_embed, a1_l)
        u_a2, _ = self.bidaf(ctx_embed, ctx_l, a2_embed, a2_l)
        u_a3, _ = self.bidaf(ctx_embed, ctx_l, a3_embed, a3_l)
        u_a4, _ = self.bidaf(ctx_embed, ctx_l, a4_embed, a4_l)
        #import ipdb; ipdb.set_trace()
        concat_a0 = torch.cat([ctx_embed, u_a0, u_q, u_a0 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a1 = torch.cat([ctx_embed, u_a1, u_q, u_a1 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a2 = torch.cat([ctx_embed, u_a2, u_q, u_a2 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a3 = torch.cat([ctx_embed, u_a3, u_q, u_a3 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a4 = torch.cat([ctx_embed, u_a4, u_q, u_a4 * ctx_embed, u_q * ctx_embed], dim=-1)

        mature_maxout_a0, _ = lstm_mature(concat_a0, ctx_l)
        mature_maxout_a1, _ = lstm_mature(concat_a1, ctx_l)
        mature_maxout_a2, _ = lstm_mature(concat_a2, ctx_l)
        mature_maxout_a3, _ = lstm_mature(concat_a3, ctx_l)
        mature_maxout_a4, _ = lstm_mature(concat_a4, ctx_l)
        #import ipdb; ipdb.set_trace()
        if self.topk == 1:
            mature_maxout_a0 = max_along_time(mature_maxout_a0, ctx_l).unsqueeze(1)
            mature_maxout_a1 = max_along_time(mature_maxout_a1, ctx_l).unsqueeze(1)
            mature_maxout_a2 = max_along_time(mature_maxout_a2, ctx_l).unsqueeze(1)
            mature_maxout_a3 = max_along_time(mature_maxout_a3, ctx_l).unsqueeze(1)
            mature_maxout_a4 = max_along_time(mature_maxout_a4, ctx_l).unsqueeze(1)
        else:
            mature_maxout_a0 = max_avg_along_time(mature_maxout_a0, ctx_l, self.topk).unsqueeze(1)
            mature_maxout_a1 = max_avg_along_time(mature_maxout_a1, ctx_l, self.topk).unsqueeze(1)
            mature_maxout_a2 = max_avg_along_time(mature_maxout_a2, ctx_l, self.topk).unsqueeze(1)
            mature_maxout_a3 = max_avg_along_time(mature_maxout_a3, ctx_l, self.topk).unsqueeze(1)
            mature_maxout_a4 = max_avg_along_time(mature_maxout_a4, ctx_l, self.topk).unsqueeze(1)

        mature_answers = torch.cat([
            mature_maxout_a0, mature_maxout_a1, mature_maxout_a2, mature_maxout_a3, mature_maxout_a4
        ], dim=1)
        out = classifier(mature_answers)  # (B, 5)
        return out

    def pointer_network():
        pass
    @staticmethod
    def get_fake_inputs(device="cuda:0"):
        bsz = 16
        q = torch.ones(bsz, 25).long().to(device)
        q_l = torch.ones(bsz).fill_(25).long().to(device)
        a = torch.ones(bsz, 5, 20).long().to(device)
        a_l = torch.ones(bsz, 5).fill_(20).long().to(device)
        a0, a1, a2, a3, a4 = [a[:, i, :] for i in range(5)]
        a0_l, a1_l, a2_l, a3_l, a4_l = [a_l[:, i] for i in range(5)]
        sub = torch.ones(bsz, 300).long().to(device)
        sub_l = torch.ones(bsz).fill_(300).long().to(device)
        vcpt = torch.ones(bsz, 300).long().to(device)
        vcpt_l = torch.ones(bsz).fill_(300).long().to(device)
        vid = torch.ones(bsz, 100, 2048).to(device)
        vid_l = torch.ones(bsz).fill_(100).long().to(device)
        return q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, a4, a4_l, sub, sub_l, vcpt, vcpt_l, vid, vid_l


if __name__ == '__main__':
    from config import BaseOptions
    import sys
    sys.argv[1:] = ["--input_streams" "sub"]
    opt = BaseOptions().parse()

    model = ABC(opt)
    model.to(opt.device)
    test_in = model.get_fake_inputs(device=opt.device)
    test_out = model(*test_in)
    print((test_out.size()))
