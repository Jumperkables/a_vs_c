######################################################
######################################################
# Models
######################################################
######################################################
# Pytorch_Lightning is a package that cleanly handles training and testing pytorch models. Check their website https://www.pytorchlightning.ai/
#class Basic(pl.LightningModule):
#    def __init__(self, args, n_answers):
#        super().__init__()
#        self.args = args
#        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#        fc_intermediate = ((n_answers-768)//2)+768
#        self.classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(768, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   # n+1 (includes unknown answer token)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.lxmert.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.lxmert.parameters():
#                param.requires_grad = False
#        self.criterion = nn.CrossEntropyLoss()
#        self.valid_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features):
#        out = self.lxmert(question, features, bboxes)[2]    #['language_output', 'vision_output', 'pooled_output']
#        out = self.classifier_fc(out)
#        return out
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        question, answer, bboxes, features, image = train_batch
#        out = self(question, bboxes, features)
#        train_loss = self.criterion(out, answer.squeeze(1))
#        out = F.softmax(out, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(out, answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        question, answer, bboxes, features, image = val_batch
#        out = self(question, bboxes, features)
#        valid_loss = self.criterion(out, answer.squeeze(1))
#        out = F.softmax(out, dim=1)
#        raise NotImplementedError("Check if you should use softmax or not use softmax for loss calculation")
#        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(out, answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        return valid_loss
#
#
#
#class BERTLSTM(pl.LightningModule):
#    def __init__(self, args, n_answers):
#        super().__init__()
#        self.args = args
#        self.bert = BertModel.from_pretrained('bert-base-uncased')       
#        self.vis_lstm = nn.LSTM(2048, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        fc_intermediate = ((n_answers-4864)//2)+4864
#        self.classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(4864, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   # n+1 (includes unknown answer token)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.bert.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.bert.parameters():
#                param.requires_grad = False
#        self.criterion = nn.CrossEntropyLoss()
#        self.valid_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features):
#        lng_out = self.bert(question)
#        lng_out = lng_out[1]
#        _, (_, vis_out) = self.vis_lstm(features)    # output, (hn, cn)
#        vis_out = vis_out.permute(1,0,2)
#        vis_out = vis_out.contiguous().view(self.args.bsz, -1)
#        combined_out = torch.cat((lng_out, vis_out), 1) # 8092
#        out = self.classifier_fc(combined_out)
#        return out
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        question, answer, bboxes, features, image = train_batch
#        out = self(question, bboxes, features)
#        train_loss = self.criterion(out, answer.squeeze(1))
#        out = F.softmax(out, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(out, answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        question, answer, bboxes, features, image = val_batch
#        out = self(question, bboxes, features)
#        valid_loss = self.criterion(out, answer.squeeze(1))
#        out = F.softmax(out, dim=1)
#        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(out, answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        return valid_loss
#
#
#
## k / (1-k) induction
#class Induction(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):
#        super().__init__()
#        self.args = args
#        if args.loss == "avsc":
#            raise NotImplementedError(f"Not implemented this with avsc loss")
#        fc_intermediate = ((n_answers-768)//2)+768
#         # High-norm / low-norm may mean high abstract/concrete. But generalised for other norms
#        self.lxmert_lownorm = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#        self.lxmert_highnorm = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#        self.low_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(768, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   #GQA has 1842 unique answers, so we pass in 1841
#        )
#        self.high_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(768, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.lxmert_highnorm.base_model.parameters():
#                param.requires_grad = False
#            for param in self.lxmert_lownorm.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.lxmert_highnorm.parameters():
#                param.requires_grad = False
#            for param in self.lxmert_lownorm.parameters():
#                param.requires_grad = False
#        self.criterion = nn.CrossEntropyLoss(reduction='none')
#        self.valid_acc = pl.metrics.Accuracy()
#        self.valid_low_acc = pl.metrics.Accuracy()
#        self.valid_high_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#        self.train_low_acc = pl.metrics.Accuracy()
#        self.train_high_acc = pl.metrics.Accuracy()
#
#        # TODO Deprecated self.idx2norm = make_idx2norm(args, ans2idx)  
#
#
#    def forward(self, question, bboxes, features):
#        out_low = self.lxmert_lownorm(question, features, bboxes)[2]    #['language_output', 'vision_output', 'pooled_output']
#        out_high = self.lxmert_highnorm(question, features, bboxes)[2]    #['language_output', 'vision_output', 'pooled_output']
#        out_low = self.low_classifier_fc(out_low)
#        out_high = self.high_classifier_fc(out_high)
#        return out_low, out_high
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        raise NotImplementedError("Check this works")
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = train_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features)
#        low_loss = self.criterion(out_low, answer.squeeze(1))
#        high_loss = self.criterion(out_high, answer.squeeze(1))
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        train_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("train_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        raise NotImplementedError("Check this works")
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = val_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features)
#        low_loss = self.criterion(out_low, answer.squeeze(1))
#        high_loss = self.criterion(out_high, answer.squeeze(1))
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        valid_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("valid_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return valid_loss
#
#
#
#
#class Hopfield_0(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
#        super().__init__()
#        self.args = args
#        # Concrete: Higher scaling beta to assert more discrete store states
#        self.high_hopfield = hpf.Hopfield(input_size = 4864, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_high, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        # Abstract: lower scaling beta to allow more metastable/global state
#        self.low_hopfield = hpf.Hopfield(input_size = 4864, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_low, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#
#        self.vis_lstm = nn.LSTM(2048, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        self.bert = BertModel.from_pretrained('bert-base-uncased')       
#        fc_intermediate = ((n_answers-1024)//2)+1024
#
#         # High-norm / low-norm may mean high abstract/concrete. But generalised for other norms
#        self.low_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   #GQA has 1842 unique answers, so we pass in 1841
#        )
#        self.high_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.bert.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.bert.parameters():
#                param.requires_grad = False
#        if args.loss == "default":
#            self.criterion = nn.CrossEntropyLoss(reduction='none')
#        elif args.loss == "avsc":
#            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
#        else:
#            raise NotImplementedError(f"Loss {args.loss} not implement for Hopfield_0 net")
#        self.valid_acc = pl.metrics.Accuracy()
#        self.valid_low_acc = pl.metrics.Accuracy()
#        self.valid_high_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#        self.train_low_acc = pl.metrics.Accuracy()
#        self.train_high_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features):
#        lng_out = self.bert(question)
#        lng_out = lng_out[1]
#        _, (_, vis_out) = self.vis_lstm(features)    # output, (hn, cn)
#        vis_out = vis_out.permute(1,0,2)
#        vis_out = vis_out.contiguous().view(self.args.bsz, -1)
#        combined_out = torch.cat((lng_out, vis_out), 1).unsqueeze(1) # 4864
#        out_low = self.low_hopfield(combined_out)
#        out_high = self.high_hopfield(combined_out)
#        out_low = out_low.squeeze(1)
#        out_high = out_high.squeeze(1)
#        out_low = self.low_classifier_fc(out_low)
#        out_high = self.high_classifier_fc(out_high)
#        return out_low, out_high
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        raise NotImplementedError("Check this work")
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = train_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        train_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("train_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        raise NotImplementedError("Check this work")
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = val_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        valid_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("valid_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return valid_loss
#
#
#
#
#class Hopfield_1(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
#        super().__init__()
#        self.args = args
#        # Concrete: Higher scaling beta to assert more discrete store states
#        self.high_hopfield = hpf.Hopfield(input_size = 4096, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_high, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        # Abstract: lower scaling beta to allow more metastable/global state
#        self.low_hopfield = hpf.Hopfield(input_size = 4096, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_low, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        #bert_config = BertConfig(hidden_size=2048, num_attention_heads=8)  # Upsize to match visual features for BiDaf
#        self.bert = BertModel.from_pretrained('bert-base-uncased')
#        self.bert_fc = nn.Linear(768, 2048)
#        self.bidaf = BidafAttn(None, method="dot")
#        self.lstm = nn.LSTM(2048, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        fc_intermediate = ((n_answers-1024)//2)+1024
#
#         # High-norm / low-norm may mean high abstract/concrete. But generalised for other norms
#        self.low_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   #GQA has 1842 unique answers, so we pass in 1841
#        )
#        self.high_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.bert.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.bert.parameters():
#                param.requires_grad = False
#        if args.loss == "default":
#            self.criterion = nn.CrossEntropyLoss(reduction='none')
#        elif args.loss == "avsc":
#            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
#        else:
#            raise NotImplementedError(f"Loss {args.loss} not implement for Hopfield_1 net")
#        self.valid_acc = pl.metrics.Accuracy()
#        self.valid_low_acc = pl.metrics.Accuracy()
#        self.valid_high_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#        self.train_low_acc = pl.metrics.Accuracy()
#        self.train_high_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features):
#        lng_out = self.bert(question)
#        lng_out = lng_out[0]
#        lng_out = self.bert_fc(lng_out)
#        lng_out_l = lng_out.shape[1]
#        features_l = features.shape[1]
#        lng_out_l = torch.LongTensor([lng_out_l]*self.args.bsz)
#        features_l = torch.LongTensor([features_l]*self.args.bsz)
#        bidaf_out = self.bidaf(lng_out, lng_out_l, features, features_l)
#        bidaf_out = bidaf_out[0]
#        _, (_, lstm_out) = self.lstm(bidaf_out) # output, (hn, cn)
#        lstm_out = lstm_out.permute(1,0,2)
#        lstm_out = lstm_out.contiguous().view(self.args.bsz, -1).unsqueeze(1)
#        out_low = self.low_hopfield(lstm_out)
#        out_high = self.high_hopfield(lstm_out)
#        out_low = out_low.squeeze(1)
#        out_high = out_high.squeeze(1)
#        out_low = self.low_classifier_fc(out_low)
#        out_high = self.high_classifier_fc(out_high)
#        return out_low, out_high
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        raise NotImplementedError("Check this works")
#        # Prepare data
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = train_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        train_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("train_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = val_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        valid_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("valid_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return valid_loss
#
#
#
#
#class Hopfield_2(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
#        super().__init__()
#        self.args = args
#        # Bert question processing
#        self.bert = BertModel.from_pretrained('bert-base-uncased')
#        self.bert_fc = nn.Linear(768, 2048)
#        # Torchvision ResNet
#        # TODO DEPRECATED??
#        #raise NotImplementedError(f"Process the h5 file for GQA and VQA-CP. Update the dataloader. ")
#        #self.img_cnn = resnet101(pretrained=True)
#        #self.img_cnn = resnet50(pretrained=True)
#        #self.img_cnn.fc = myutils.Identity() # Really cool trick, read myutils for explanation
#        #for param in self.img_cnn.parameters():
#        #    param.requires_grad = False
#        # Concrete: Higher scaling beta to assert more discrete store states
#        self.high_hopfield = hpf.Hopfield(input_size = 4096, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_high, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        self.high_bidaf = BidafAttn(None, method="dot")
#        self.high_lstm = nn.LSTM(2048, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        # Abstract: lower scaling beta to allow more metastable/global state
#        self.low_hopfield = hpf.Hopfield(input_size = 4096, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_low, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        self.low_bidaf = BidafAttn(None, method="dot")
#        self.low_lstm = nn.LSTM(2048, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        fc_intermediate = ((n_answers-1024)//2)+1024
#
#         # High-norm / low-norm may mean high abstract/concrete. But generalised for other norms
#        self.low_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   #GQA has 1842 unique answers, so we pass in 1841
#        )
#        self.high_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.bert.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.bert.parameters():
#                param.requires_grad = False
#        if args.loss == "default":
#            self.criterion = nn.CrossEntropyLoss(reduction='none')
#        elif args.loss == "avsc":
#            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
#        else:
#            raise NotImplementedError(f"Loss {args.loss} not implement for Hopfield_2 net")
#        self.valid_acc = pl.metrics.Accuracy()
#        self.valid_low_acc = pl.metrics.Accuracy()
#        self.valid_high_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#        self.train_low_acc = pl.metrics.Accuracy()
#        self.train_high_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features, image):
#        # Process language
#        lng_out = self.bert(question)
#        lng_out = lng_out[0]
#        lng_out = self.bert_fc(lng_out)
#        lng_out_l = lng_out.shape[1]
#        # Process image
#        # TODO Deprecated? image_feat = self.img_cnn(image).unsqueeze(1)
#        image_feat = image.unsqueeze(1)
#        image_l = torch.LongTensor([1]*self.args.bsz)
#        #Get lengths
#        features_l = features.shape[1]
#        lng_out_l = torch.LongTensor([lng_out_l]*self.args.bsz)
#        features_l = torch.LongTensor([features_l]*self.args.bsz)
#        # High stream (objects)
#        high_bidaf_out = self.high_bidaf(lng_out, lng_out_l, features, features_l)
#        high_bidaf_out = high_bidaf_out[0]
#        _, (_, high_lstm_out) = self.high_lstm(high_bidaf_out) # output, (hn, cn)
#        high_lstm_out = high_lstm_out.permute(1,0,2)
#        high_lstm_out = high_lstm_out.contiguous().view(self.args.bsz, -1).unsqueeze(1)
#        # Low Stream (image)
#        low_bidaf_out = self.low_bidaf(lng_out, lng_out_l, image_feat, image_l)
#        low_bidaf_out = low_bidaf_out[0]
#        _, (_, low_lstm_out) = self.low_lstm(low_bidaf_out)
#        low_lstm_out = low_lstm_out.permute(1,0,2)
#        low_lstm_out = low_lstm_out.contiguous().view(self.args.bsz, -1).unsqueeze(1)
#        # Hopfields and FC
#        out_low = self.low_hopfield(low_lstm_out)
#        out_high = self.high_hopfield(high_lstm_out)
#        out_low = out_low.squeeze(1)
#        out_high = out_high.squeeze(1)
#        out_low = self.low_classifier_fc(out_low)
#        out_high = self.high_classifier_fc(out_high)
#        return out_low, out_high
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = train_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        train_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("train_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        raise NotImplementedError("Check this is implemented properly")
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = val_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        valid_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("valid_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return valid_loss
#
#
#
#
#class Hopfield_3(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
#        super().__init__()
#        self.args = args
#        # LXMERT Models
#        self.high_lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#        self.low_lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
#        # Language/Vision LSTM
#        self.lng_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        self.vis_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
#        # Hopfield Nets
#        self.high_hopfield = hpf.Hopfield(input_size = 8960, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_high, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        self.low_hopfield = hpf.Hopfield(input_size = 8960, hidden_size = 1024, output_size = 1024, pattern_size = 1, num_heads = 7, scaling = args.hopfield_beta_low, update_steps_max = 3, update_steps_eps = 1e-4, dropout = 0.2)
#        fc_intermediate = ((n_answers-1024)//2)+1024
#         # High-norm / low-norm may mean high abstract/concrete. But generalised for other norms
#        self.low_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)   #GQA has 1842 unique answers, so we pass in 1841
#        )
#        self.high_classifier_fc = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(1024, fc_intermediate),
#            nn.BatchNorm1d(fc_intermediate),
#            nn.GELU(),
#            nn.Dropout(0.2),
#            nn.Linear(fc_intermediate, n_answers+1)
#        )
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for param in self.high_lxmert.base_model.parameters():
#                param.requires_grad = False
#            for param in self.low_lxmert.base_model.parameters():
#                param.requires_grad = False
#        elif args.unfreeze == "none":
#            for param in self.high_lxmert.parameters():
#                param.requires_grad = False
#            for param in self.low_lxmert.parameters():
#                param.requires_grad = False
#        if args.loss == "default":
#            self.criterion = nn.CrossEntropyLoss(reduction='none')
#        elif args.loss == "avsc":
#            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
#        else:
#            raise NotImplementedError(f"Loss {args.loss} not implement for Hopfield_3 net")
#        self.valid_acc = pl.metrics.Accuracy()
#        self.valid_low_acc = pl.metrics.Accuracy()
#        self.valid_high_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#        self.train_low_acc = pl.metrics.Accuracy()
#        self.train_high_acc = pl.metrics.Accuracy()
#
#
#    def forward(self, question, bboxes, features, image):
#        # Process language
#        out_low = self.low_lxmert(question, features, bboxes)       #['language_output', 'vision_output', 'pooled_output']
#        lng_out_low, vis_out_low, x_out_low = out_low['language_output'], out_low['vision_output'], out_low['pooled_output']
#        out_high = self.high_lxmert(question, features, bboxes)     #['language_output', 'vision_output', 'pooled_output']
#        lng_out_high, vis_out_high, x_out_high = out_high['language_output'], out_high['vision_output'], out_high['pooled_output']
#        # x stands for 'cross', see naming scheme in documentation
#        # Language/Vision LSTM processing
#        _, (_, lng_out_low) = self.lng_lstm(lng_out_low)
#        _, (_, lng_out_high) = self.lng_lstm(lng_out_high)
#        _, (_, vis_out_low) = self.vis_lstm(vis_out_low)
#        _, (_, vis_out_high) = self.vis_lstm(vis_out_high)
#        lng_out_low = lng_out_low.permute(1,0,2).contiguous().view(self.args.bsz, -1)
#        lng_out_high = lng_out_high.permute(1,0,2).contiguous().view(self.args.bsz, -1)
#        vis_out_low = vis_out_low.permute(1,0,2).contiguous().view(self.args.bsz, -1)
#        vis_out_high = vis_out_high.permute(1,0,2).contiguous().view(self.args.bsz, -1)
#        # Hopfield
#        out_low = torch.cat((lng_out_low, vis_out_low, x_out_low), dim=1).unsqueeze(1)
#        out_high = torch.cat((lng_out_high, vis_out_high, x_out_high), dim=1).unsqueeze(1)
#        out_low = self.low_hopfield(out_low)
#        out_high = self.high_hopfield(out_high)
#        out_low = out_low.squeeze(1)
#        out_high = out_high.squeeze(1)
#        out_low = self.low_classifier_fc(out_low)
#        out_high = self.high_classifier_fc(out_high)
#        return out_low, out_high
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = train_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        train_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("train_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        raise NotImplementedError("Check that this works")
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens = val_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        valid_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("valid_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return valid_loss

#class Dual_LxForQA(pl.LightningModule):
#    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
#        super().__init__()
#        self.args = args
#        # LXMERT Models
#        dummy_conf = Dummy_Lxmert_Conf(hidden_size=768)
#        if args.dataset == "GQA":
#            high_ans_head = LxmertVisualAnswerHead(config=dummy_conf, num_labels=len(ans2idx))
#            low_ans_head = LxmertVisualAnswerHead(config=dummy_conf, num_labels=len(ans2idx))
#        elif args.dataset in ["VQACP","VQACP2"]:
#            high_ans_head = LxmertVisualAnswerHead(config=dummy_conf, num_labels=len(ans2idx)+1)
#            low_ans_head = LxmertVisualAnswerHead(config=dummy_conf, num_labels=len(ans2idx)+1)
#        self.high_lxmert = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased")
#        self.low_lxmert = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased")
#        self.high_lxmert.answer_head = high_ans_head
#        self.low_lxmert.answer_head = low_ans_head
#        for name, param in self.high_lxmert.named_parameters():
#            param.requires_grad = True
#        for name, param in self.low_lxmert.named_parameters():
#            param.requires_grad = True
#        if args.unfreeze == "all":
#            pass
#        elif args.unfreeze == "heads":
#            for name, param in self.high_lxmert.named_parameters():
#                if not("attention" in name):
#                    param.requires_grad = False
#            for name, param in self.low_lxmert.named_parameters():
#                if not("attention" in name):
#                    param.requires_grad = False
#        elif args.unfreeze == "qa_head":
#            for name, param in self.high_lxmert.named_parameters():
#                if not("answer_head" in name):
#                    param.requires_grad = False
#            for name, param in self.low_lxmert.named_parameters():
#                if not("answer_head" in name):
#                    param.requires_grad = False
#        elif args.unfreeze == "none":
#            for name, param in self.high_lxmert.named_parameters():
#                param.requires_grad = False
#            for name, param in self.low_lxmert.named_parameters():
#                param.requires_grad = False
#        if args.loss == "default":
#            self.criterion = nn.CrossEntropyLoss(reduction='none')
#        elif args.loss == "avsc":
#            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
#        else:
#            raise NotImplementedError(f"Loss {args.loss} not implement for Hopfield_3 net")
#        self.valid_acc = pl.metrics.Accuracy()
#        self.valid_low_acc = pl.metrics.Accuracy()
#        self.valid_high_acc = pl.metrics.Accuracy()
#        self.train_acc = pl.metrics.Accuracy()
#        self.train_low_acc = pl.metrics.Accuracy()
#        self.train_high_acc = pl.metrics.Accuracy()
#                
#
#    def forward(self, question, bboxes, features, image):
#        # Process language
#        out_low = self.low_lxmert(question, features, bboxes)['question_answering_score']
#        out_high = self.high_lxmert(question, features, bboxes)['question_answering_score']
#        return out_low, out_high
#
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#        return optimizer
#
#    def training_step(self, train_batch, batch_idx):
#        # Prepare data
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _ = train_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        train_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("train_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, val_batch, batch_idx):
#        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _ = val_batch
#        high_norms = return_norm
#        low_norms = torch.ones(len(high_norms)).to(self.device)
#        low_norms = low_norms - high_norms
#        out_low, out_high = self(question, bboxes, features, image)
#        if self.args.loss == "default":
#            low_loss = self.criterion(out_low, answer.squeeze(1))
#            high_loss = self.criterion(out_high, answer.squeeze(1))
#        elif self.args.loss == "avsc":
#            low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
#            high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
#        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
#        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
#        valid_loss = low_loss + high_loss
#        out_high = F.softmax(out_high, dim=1)
#        out_low = F.softmax(out_low, dim=1)
#        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_loss", low_loss, on_step=False, on_epoch=True)
#        self.log("valid_high_loss", high_loss, on_step=False, on_epoch=True)
#        self.log("valid_acc", self.valid_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=True, on_step=False, on_epoch=True)
#        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
#        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
#        return valid_loss


class Dual_LxLSTM(pl.LightningModule):
    def __init__(self, args, n_answers, ans2idx):   # Pass ans2idx from relevant dataset object
        super().__init__()
        self.args = args
        # LXMERT Models
        self.high_lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.low_lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        # Language/Vision LSTM
        self.lng_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.vis_lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        fc_intermediate = ((n_answers-8960)//2)+8960
        self.low_classifier_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(8960, fc_intermediate),
            nn.BatchNorm1d(fc_intermediate),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fc_intermediate, n_answers+1)   #GQA has 1842 unique answers, so we pass in 1841
        )
        self.high_classifier_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(8960, fc_intermediate),
            nn.BatchNorm1d(fc_intermediate),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fc_intermediate, n_answers+1)
        )
        for name, param in self.high_lxmert.named_parameters():
            param.requires_grad = True
        for name, param in self.low_lxmert.named_parameters():
            param.requires_grad = True
        if args.unfreeze == "all":
            pass
        elif args.unfreeze == "heads":
            for name, param in self.high_lxmert.named_parameters():
                if not("attention" in name):
                    param.requires_grad = False
            for name, param in self.low_lxmert.named_parameters():
                if not("attention" in name):
                    param.requires_grad = False
        elif args.unfreeze == "none":
            for name, param in self.high_lxmert.named_parameters():
                param.requires_grad = False
            for name, param in self.low_lxmert.named_parameters():
                param.requires_grad = False
        if args.loss == "default":
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif args.loss == "avsc":
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError(f"Loss {args.loss} not implement for {args.model} net")
        # Logging for metrics
        self.valid_acc = torchmetrics.Accuracy()
        self.valid_low_acc = torchmetrics.Accuracy()
        self.valid_high_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()
        self.train_low_acc = torchmetrics.Accuracy()
        self.train_high_acc = torchmetrics.Accuracy()
        self.best_acc = 0
        self.predictions = {}
        self.high_predictions = {}
        self.high_attentions = {}
        self.low_predictions = {}
        self.low_attentions = {}
        # TODO DEPRECATED# Correct answer ids to work out consistency/plausability etc..
        # self.correct_answers = []
        # self.correct_answers_low = []
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
                nn.Linear(fc_intermediate, n_answers+1)
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
        out_low = self.low_lxmert(question, features, bboxes, output_attentions=True)
        # NOTE According to GitHub issue: https://github.com/airsplay/lxmert/issues/26, bounding boxes are of form (x0,y0,x1,y1) for lxmert
        #['language_output', 'vision_output', 'pooled_output', 'language_attentions', 'vision_attentions', 'cross_attentions']
        lng_out_low, vis_out_low, x_out_low = out_low['language_output'], out_low['vision_output'], out_low['pooled_output']
        vis_attns_low = torch.stack(out_low['vision_attentions']).mean(dim=0).mean(dim=1)
        out_high = self.high_lxmert(question, features, bboxes, output_attentions=True)
        #['language_output', 'vision_output', 'pooled_output', 'language_attentions', 'vision_attentions', 'cross_attentions']
        lng_out_high, vis_out_high, x_out_high = out_high['language_output'], out_high['vision_output'], out_high['pooled_output']
        vis_attns_high = torch.stack(out_high['vision_attentions']).mean(dim=0).mean(dim=1)
        if self.args.rubi == "rubi":
            ## Language only
            #TODO deprecated? lng_out_biased = self.biased_bert(question)[0]
            #_, (_, lng_out_biased) = self.biased_lng_lstm(lng_out_biased)
            #lng_out_biased = lng_out_biased.permute(1,0,2).contiguous().view(self.args.bsz, -1)
            lng_out_biased = self.biased_bert(question)[1].squeeze(1)
            out_biased = self.biased_classifier_fc(lng_out_biased)
        else:
            out_biased = None
        # x stands for 'cross', see naming scheme in documentation
        # Language/Vision LSTM processing
        _, (_, lng_out_low) = self.lng_lstm(lng_out_low)
        _, (_, lng_out_high) = self.lng_lstm(lng_out_high)
        _, (_, vis_out_low) = self.vis_lstm(vis_out_low)
        _, (_, vis_out_high) = self.vis_lstm(vis_out_high)
        bsz = lng_out_low.shape[1]
        lng_out_low = lng_out_low.permute(1,0,2).contiguous().view(bsz, -1)
        lng_out_high = lng_out_high.permute(1,0,2).contiguous().view(bsz, -1)
        vis_out_low = vis_out_low.permute(1,0,2).contiguous().view(bsz, -1)
        vis_out_high = vis_out_high.permute(1,0,2).contiguous().view(bsz, -1)
        out_low = torch.cat((lng_out_low, vis_out_low, x_out_low), dim=1)
        out_high = torch.cat((lng_out_high, vis_out_high, x_out_high), dim=1)
        out_low = self.low_classifier_fc(out_low)
        out_high = self.high_classifier_fc(out_high)
        return out_low, out_high, out_biased, vis_attns_low, vis_attns_high

    def configure_optimizers(self):
        other_optimizer = torch.optim.Adam(nn.ParameterList([p for n,p in self.named_parameters() if "lxmert" not in n]), lr=self.args.lr)
        lxmert_optimizer = torch.optim.Adam(nn.ParameterList([p for n,p in self.named_parameters() if "lxmert" in n]), lr=self.args.lr/5)
        return other_optimizer, lxmert_optimizer

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        # Prepare data
        other_optimizer, lxmert_optimizer = self.optimizers()
        other_optimizer.zero_grad()
        lxmert_optimizer.zero_grad()
        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _, q_id_ret, _ = train_batch
        out_low, out_high, out_biased, vis_attns_low, vis_attns_high = self(question, bboxes, features, image) # out_biased is from potential RUBi outputs
        if self.args.dual_loss_style == "linear":
            high_norms = return_norm
            low_norms = torch.ones(len(high_norms)).to(self.device)
            low_norms = low_norms - high_norms
        elif self.args.dual_loss_style == "quadr":
            high_norms = (return_norm)**2
            low_norms = (return_norm-1)**2
        elif self.args.dual_loss_style == "cubic":
            high_norms = (return_norm)**3
            low_norms = -1*((return_norm-1)**3)
        elif self.args.dual_loss_style == "4th":
            high_norms = (return_norm)**4
            low_norms = (return_norm-1)**4
        else:
            raise NotImplementedError(f"`{self.args.dual_loss_style}` not implemented")
        if self.args.loss == "default":
            if self.args.rubi == "rubi":
                #['combined_loss']
                #['main_loss']
                #['biased_loss']
                low_loss = self.criterion(out_low, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                low_biased_loss = low_loss['biased_loss']
                low_loss = low_loss['combined_loss']+low_biased_loss
                high_loss = self.criterion(out_high, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                low_loss = self.criterion(out_low, answer.squeeze(1))
                high_loss = self.criterion(out_high, answer.squeeze(1))
        elif self.args.loss == "avsc":
            if self.args.rubi == "rubi":
                low_loss = self.criterion(out_high, out_biased, abs_answer_tens, biased_loss_weighting=1.0)
                low_biased_loss = low_loss['biased_loss']
                low_loss = low_loss['combined_loss']+low_biased_loss
                low_loss = torch.mean(low_loss, 1)
                high_loss = self.criterion(out_high, out_biased, conc_answer_tens, biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
                high_loss = torch.mean(high_loss, 1)
            else:
                low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
                high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
        train_loss = low_loss + high_loss
        out_high = F.softmax(out_high, dim=1)
        out_low = F.softmax(out_low, dim=1)
        self.log("train_loss", train_loss, prog_bar=False, on_step=True)#, on_epoch=True)
        self.log("train_low_loss", low_loss, on_step=True)#, on_epoch=True)
        self.log("train_high_loss", high_loss, on_step=True)#False, on_epoch=True)
        self.log("train_acc", self.train_acc(F.softmax(out_high+out_low, dim=1), answer.squeeze(1)), prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_low_acc", self.train_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
        self.log("train_high_acc", self.train_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
        if self.args.rubi == "rubi":
            self.log("train_low_biased_loss", low_biased_loss, on_step=True)#False, on_epoch=True)
            self.log("train_high_biased_loss", high_biased_loss, on_step=True)#False, on_epoch=True)
        self.manual_backward(train_loss)
        other_optimizer.step()
        lxmert_optimizer.step()
        #return train_loss

    def validation_step(self, val_batch, batch_idx):
        question, answer, bboxes, features, image, return_norm, abs_answer_tens, conc_answer_tens, _, q_id_ret, img_dims = val_batch
        out_low, out_high, out_biased, vis_attns_low, vis_attns_high = self(question, bboxes, features, image)
        if self.args.dual_loss_style == "linear":
            high_norms = return_norm
            low_norms = torch.ones(len(high_norms)).to(self.device)
            low_norms = low_norms - high_norms
        elif self.args.dual_loss_style == "quadr":
            high_norms = (return_norm)**2
            low_norms = (return_norm-1)**2
        elif self.args.dual_loss_style == "cubic":
            high_norms = (return_norm)**3
            low_norms = -1*((return_norm-1)**3)
        elif self.args.dual_loss_style == "4th":
            high_norms = (return_norm)**4
            low_norms = (return_norm-1)**4
        else:
            raise NotImplementedError(f"`{self.args.dual_loss_style}` not implemented")
        if self.args.loss == "default":
            if self.args.rubi == "rubi":
                #['combined_loss']
                #['main_loss']
                #['biased_loss']
                low_loss = self.criterion(out_low, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                low_biased_loss = low_loss['biased_loss']
                low_loss = low_loss['combined_loss']+low_biased_loss
                high_loss = self.criterion(out_high, out_biased, answer.squeeze(1), biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
            else:
                low_loss = self.criterion(out_low, answer.squeeze(1))
                high_loss = self.criterion(out_high, answer.squeeze(1))
        elif self.args.loss == "avsc":
            if self.args.rubi == "rubi":
                low_loss = self.criterion(out_high, out_biased, abs_answer_tens, biased_loss_weighting=1.0)
                low_biased_loss = low_loss['biased_loss']
                low_loss = low_loss['combined_loss']+low_biased_loss
                low_loss = torch.mean(low_loss, 1)
                high_loss = self.criterion(out_high, out_biased, conc_answer_tens, biased_loss_weighting=1.0)
                high_biased_loss = high_loss['biased_loss']
                high_loss = high_loss['combined_loss']+high_biased_loss
                high_loss = torch.mean(high_loss, 1)
            else:
                low_loss = torch.mean(self.criterion(out_low, abs_answer_tens), 1)
                high_loss = torch.mean(self.criterion(out_high, conc_answer_tens), 1)
        low_loss = torch.dot(low_norms, low_loss) / len(low_loss)
        high_loss = torch.dot(high_norms, high_loss) / len(high_loss)
        valid_loss = low_loss + high_loss
        out_high = F.softmax(out_high, dim=1)
        out_low = F.softmax(out_low, dim=1)
        combined = F.softmax(out_high+out_low, dim=1)
        self.log("valid_loss", valid_loss, prog_bar=False, on_step=True)#False, on_epoch=True)
        self.log("valid_low_loss", low_loss, on_step=True)#False, on_epoch=True)
        self.log("valid_high_loss", high_loss, on_step=True)#False, on_epoch=True)
        self.log("valid_acc", self.valid_acc(combined, answer.squeeze(1)), prog_bar=False, on_step=False, on_epoch=True)
        self.log("valid_low_acc", self.valid_acc(out_low, answer.squeeze(1)), on_step=False, on_epoch=True)
        self.log("valid_high_acc", self.valid_acc(out_high, answer.squeeze(1)), on_step=False, on_epoch=True)
        combined = combined.argmax(dim=1)
        out_high = out_high.argmax(dim=1)
        out_low = out_low.argmax(dim=1)
        # Move the rescaling to the forward pass
        vis_attns_high = vis_attns_high.mean(dim=1)
        vis_attns_low = vis_attns_low.mean(dim=1)
        #vis_attns_high.cpu()
        #vis_attns_low.cpu()
        #bboxes.cpu()
        for i in range(len(q_id_ret)):
            q_idx = f"{q_id_ret[i][0]}".zfill(q_id_ret[i][1])
            self.predictions[q_idx] = self.val_dataloader.dataloader.dataset.idx2ans[int(combined[i])]
            self.high_predictions[q_idx] = self.val_dataloader.dataloader.dataset.idx2ans[int(out_high[i])]
            self.high_attentions[q_idx] = [((float(bboxes[i][j][0]), float(bboxes[i][j][1]), float(bboxes[i][j][2]), float(bboxes[i][j][3])), float(vis_attns_high[i][j])) for j in range(len(vis_attns_high[i]))]
            self.low_predictions[q_idx] = self.val_dataloader.dataloader.dataset.idx2ans[int(out_low[i])]
            self.low_attentions[q_idx] = [((float(bboxes[i][j][0]), float(bboxes[i][j][1]), float(bboxes[i][j][2]), float(bboxes[i][j][3])), float(vis_attns_low[i][j])) for j in range(len(vis_attns_low[i]))]
        if self.args.rubi == "rubi":
            self.log("valid_low_biased_loss", low_biased_loss, on_step=True)#False, on_epoch=True)
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
                myutils.save_json(self.predictions, os.path.join(metrics_dir, "predictions.json"))
                myutils.save_json(self.high_predictions, os.path.join(metrics_dir, "high_predictions.json"))
                myutils.save_json(self.low_predictions, os.path.join(metrics_dir, "low_predictions.json"))
                myutils.save_json(self.high_attentions, os.path.join(metrics_dir, "high_attentions.json"))
                myutils.save_json(self.low_attentions, os.path.join(metrics_dir, "low_attentions.json"))
                # Plot 'predictions.json' without attention
                if self.args.dataset == "GQA":
                    val_questions = "val_balanced_questions.json"
                elif self.args.dataset == "GQA-ABSMIXED":
                    val_questions = "absMixed_val_questions.json"
                os.system(f"python eval.py --tier 'val' --checkpoint_path 'checkpoints/{args.jobname}' --score_file_name 'scores.txt' --scenes 'val_sceneGraphs.json' --questions '{val_questions}' --choices 'val_choices.json' --predictions 'predictions.json' --consistency")
                with open(os.path.join(metrics_dir, "scores.txt")) as f:
                    scores = f.read().replace('\n', '<br />')
                    scores = "<p>"+scores+"</p>"
                    self.log("scores", wandb.Html(scores))
                # Plot 'high_predictions.json' with high attentions
                os.system(f"python eval.py --tier 'val' --checkpoint_path 'checkpoints/{args.jobname}' --score_file_name 'high_scores.txt' --scenes 'val_sceneGraphs.json' --questions '{val_questions}' --choices 'val_choices.json' --predictions 'high_predictions.json' --attentions 'high_attentions.json' --consistency --grounding --objectFeatures")
                with open(os.path.join(metrics_dir, "high_scores.txt")) as f:
                    scores = f.read().replace('\n', '<br />')
                    scores = "<p>"+scores+"</p>"
                    self.log("high_scores", wandb.Html(scores))
                # Plot 'low_predictions.json' with low attentions
                os.system(f"python eval.py --tier 'val' --checkpoint_path 'checkpoints/{args.jobname}' --score_file_name 'low_scores.txt' --scenes 'val_sceneGraphs.json' --questions '{val_questions}' --choices 'val_choices.json' --predictions 'low_predictions.json' --attentions 'low_attentions.json' --consistency --grounding --objectFeatures")
                with open(os.path.join(metrics_dir, "low_scores.txt")) as f:
                    scores = f.read().replace('\n', '<br />')
                    scores = "<p>"+scores+"</p>"
                    self.log("low_scores", wandb.Html(scores))



class Dummy_Lxmert_Conf():
    # Just to pass hidden_size to LxmertVisualAnswerHead
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


