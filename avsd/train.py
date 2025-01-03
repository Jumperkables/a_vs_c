import argparse
import datetime
import gc
import math
import os
import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataloader import VisDialDataset
from encoders import Encoder, LateFusionEncoder
from decoders import Decoder

from utils import process_ranks, scores_to_ranks, get_gt_ranks

import wandb

import warnings
warnings.filterwarnings("ignore") 

parser = argparse.ArgumentParser()
VisDialDataset.add_cmdline_args(parser)
LateFusionEncoder.add_cmdline_args(parser)

parser.add_argument_group('Input modalites arguments')
parser.add_argument('--input_type', default='question_dialog_video_audio', choices=['question_only',
                                                                     'question_dialog',
                                                                     'question_audio',
                                                                     'question_image',
                                                                     'question_video',
                                                                     'question_caption_image',
                                                                     'question_dialog_video',
                                                                     'question_dialog_image',
                                                                     'question_video_audio',
                                                                     'question_dialog_video_audio'], help='Specify the inputs')

parser.add_argument_group('Encoder Decoder choice arguments')
parser.add_argument('--encoder', default='lf-ques-im-hist', choices=['lf-ques-im-hist'], help='Encoder to use for training')
parser.add_argument('--concat_history', default=True, help='True for lf encoding')
parser.add_argument('--decoder', default='disc', choices=['disc'], help='Decoder to use for training')

parser.add_argument_group('Optimization related arguments')
parser.add_argument("--jobname", default="default", help="Unique name ID of the job")
parser.add_argument('--num_epochs', default=20, type=int, help='Epochs')
parser.add_argument('--batch_size', default=12, type=int, help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--lr_decay_rate', default=0.9997592083, type=float, help='Decay for lr')
parser.add_argument('--min_lr', default=5e-5, type=float, help='Minimum learning rate')
parser.add_argument('--weight_init', default='xavier', choices=['xavier', 'kaiming'], help='Weight initialization strategy')
parser.add_argument('--weight_decay', default=0.00075, help='Weight decay for l2 regularization')
parser.add_argument('--overfit', action='store_true', help='Overfit on 5 examples, meant for debugging')
parser.add_argument('--gpuid', default=0, type=int, help='GPU id to use')

parser.add_argument_group('Checkpointing related arguments')
parser.add_argument('--load_path', default='', help='Checkpoint to load path from')
parser.add_argument('--save_path', default='.results', help='Path to save checkpoints')
parser.add_argument('--save_step', default=2, type=int, help='Save checkpoint after every save_step epochs')
parser.add_argument("--log", action="store_true", help="Use weights and biases logging")

parser.add_argument_group('Word norm alterations')
parser.add_argument("--mrc_norms_conditions", type=str, nargs="+", help="The conditions on each norm as boolean. e.g. conc-lt-500 is concreteness less than 500 words kept. Operations: lt, gt =  less-than, greaters-than; eq, neq = equals, not-equals. Each listed condition is applied, i.e. boolean and of all conditions")

from pathlib import Path
def resolve_path(path):
    """
    Resolve the relative path of this file
    Inputs:
        path: path to the file relative to this file
    """
    return((Path(__file__).parent / path).resolve())


# ----------------------------------------------------------------------------
# input arguments and options
# ----------------------------------------------------------------------------

args = parser.parse_args()
args.load_flag = (args.load_path != '')
args.save_path = resolve_path(args.save_path)
args.load_path = resolve_path(args.load_path)

if args.log:
    wandb.init(project="a_vs_c", name=args.jobname)
    wandb.config.update(args)

start_time = datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')
if args.save_path == 'checkpoints/':
    args.save_path += start_time

# seed for reproducibility
torch.manual_seed(1234)

# set device and default tensor type
if args.gpuid >= 0:
    torch.cuda.manual_seed_all(1234)
    torch.cuda.set_device(args.gpuid)

# transfer all options to model
model_args = args

# ----------------------------------------------------------------------------
# read saved model and args
# ----------------------------------------------------------------------------

if args.load_flag:
    components = torch.load(args.load_path)
    model_args = components['model_args']
    model_args.gpuid = args.gpuid
    model_args.batch_size = args.batch_size
else:
    args.save_path = resolve_path(args.save_path)
    args.load_path = resolve_path(args.load_path)


    # this is required by dataloader
    args.img_norm = model_args.img_norm

for arg in vars(args):
    print('{:<20}: {}'.format(arg, getattr(args, arg)))

# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------

dataset = VisDialDataset(args, ['train'])
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=dataset.collate_fn)

dataset_val = VisDialDataset(args, ['val'])
dataloader_val = DataLoader(dataset_val,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=dataset.collate_fn)

# ----------------------------------------------------------------------------
# setting model args
# ----------------------------------------------------------------------------

# transfer some useful args from dataloader to model
for key in {'num_data_points', 'vocab_size', 'max_ques_count',
            'max_ques_len', 'max_ans_len'}:
    setattr(model_args, key, getattr(dataset, key))

# iterations per epoch
setattr(args, 'iter_per_epoch', math.ceil(dataset.num_data_points['train'] / args.batch_size))
print("{} iter per epoch.".format(args.iter_per_epoch))

# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------

encoder = Encoder(model_args)
decoder = Decoder(model_args, encoder)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay_rate)

if args.load_flag:
    encoder.load_state_dict(components['encoder'])
    decoder.load_state_dict(components['decoder'])
    print("Loaded model from {}".format(args.load_path))
print("Encoder: {}".format(args.encoder))
print("Decoder: {}".format(args.decoder))

if args.gpuid >= 0:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    criterion = criterion.cuda()

# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

encoder.train()
decoder.train()


os.makedirs(args.save_path, exist_ok=True)

running_loss = 0.0
train_begin = datetime.datetime.utcnow()
print("Training start time: {}".format(datetime.datetime.strftime(train_begin, '%d-%b-%Y-%H:%M:%S')))

log_loss = []
for epoch in range(1, model_args.num_epochs + 1):
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = Variable(batch[key])
                if args.gpuid >= 0:
                    batch[key] = batch[key].cuda()

        # --------------------------------------------------------------------
        # forward-backward pass and optimizer step
        # --------------------------------------------------------------------
        enc_out = encoder(batch)
        dec_out = decoder(enc_out, batch)
        cur_loss = criterion(dec_out, batch['ans_ind'].view(-1))
        cur_loss.backward()

        optimizer.step()
        gc.collect()

        # --------------------------------------------------------------------
        # update running loss and decay learning rates
        # --------------------------------------------------------------------
        train_loss = cur_loss.item()
        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * cur_loss.item()
        else:
            running_loss = cur_loss.item()

        if optimizer.param_groups[0]['lr'] > args.min_lr:
            scheduler.step()


        # --------------------------------------------------------------------
        # print after every few iterations
        # --------------------------------------------------------------------
        if i % 100 == 0:
            validation_losses = []
            all_ranks = []
            for _, val_batch in enumerate(dataloader_val):
                for key in val_batch:
                    if not isinstance(val_batch[key], list):
                        val_batch[key] = Variable(val_batch[key])
                        if args.gpuid >= 0:
                            val_batch[key] = val_batch[key].cuda()
                enc_out = encoder(val_batch)
                dec_out = decoder(enc_out, val_batch)
                ####
                ranks = scores_to_ranks(dec_out.data)
                gt_ranks = get_gt_ranks(ranks, val_batch['ans_ind'].data)
                all_ranks.append(gt_ranks)
                ####
                cur_loss = criterion(dec_out, val_batch['ans_ind'].view(-1))
                validation_losses.append(cur_loss.item())

            all_ranks = torch.cat(all_ranks, 0)
            metric_dict = process_ranks(all_ranks)
            validation_loss = np.mean(validation_losses)

            iteration = (epoch - 1) * args.iter_per_epoch + i

            log_loss.append((epoch,
                             iteration,
                             running_loss,
                             train_loss,
                             validation_loss,
                             optimizer.param_groups[0]['lr']))

            # print current time, running average, learning rate, iteration, epoch
            print("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][val loss: {:6f}][lr: {:7f}]".format(
                datetime.datetime.utcnow() - train_begin, epoch,
                    iteration, running_loss, validation_loss,
                    optimizer.param_groups[0]['lr']))
            if args.log:
                wandb.log({
                    "Train Loss": running_loss,
                    "Val Loss": validation_loss,
                    "r@1":      metric_dict["r@1"],
                    "r@5":      metric_dict["r@5"],
                    "r@10":     metric_dict["r@10"],
                    "meanR":    metric_dict["meanR"],
                    "meanRR":   metric_dict["meanRR"],
                    "Epoch": epoch
                    })

    # ------------------------------------------------------------------------
    # save checkpoints and final model
    # ------------------------------------------------------------------------
    if epoch % args.save_step == 0:
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': encoder.args
        }, os.path.join(args.save_path, '{}.pth'.format(epoch)))

torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': encoder.args
}, os.path.join(args.save_path, 'model_final.pth'))

np.save(os.path.join(args.save_path, 'log_loss'), log_loss)

