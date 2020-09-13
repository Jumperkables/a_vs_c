import argparse
import datetime
import gc
import json
import math
import os
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import VisDialDataset
from encoders import Encoder, LateFusionEncoder
from decoders import Decoder
from utils import process_ranks, scores_to_ranks, get_gt_ranks

import wandb

import warnings
warnings.filterwarnings("ignore") 

from pathlib import Path
def resolve_path(path):
    """
    Resolve the relative path of this file
    Inputs:
        path: path to the file relative to this file
    """
    return((Path(__file__).parent / path).resolve())


parser = argparse.ArgumentParser()
VisDialDataset.add_cmdline_args(parser)
LateFusionEncoder.add_cmdline_args(parser)

parser.add_argument('-input_type', default='question_dialog_video_audio', choices=['question_only',
                                                                     'question_dialog',
                                                                     'question_audio',
                                                                     'question_image',
                                                                     'question_video',
                                                                     'question_caption_image',
                                                                     'question_dialog_video',
                                                                     'question_dialog_image',
                                                                     'question_video_audio',
                                                                     'question_dialog_video_audio'], help='Specify the inputs')

parser.add_argument_group('Evaluation related arguments')
parser.add_argument('--jobname', default="default", help="Unique name of these models")
parser.add_argument('-load_path', default='.results/REPLACEME', help='Checkpoint to load path from')
parser.add_argument('-split', default='test', choices=['val', 'test', 'train'], help='Split to evaluate on')
parser.add_argument('-use_gt', action='store_true', help='Whether to use ground truth for retrieving ranks')
parser.add_argument('-batch_size', default=12, type=int, help='Batch size')
parser.add_argument('-gpuid', default=0, type=int, help='GPU id to use')
parser.add_argument('-overfit', action='store_true', help='Use a batch of only 5 examples, useful for debugging')

parser.add_argument_group('Submission related arguments')
parser.add_argument('-save_ranks', action='store_true', help='Whether to save retrieved ranks')
parser.add_argument('-save_path', default='logs/ranks.json', help='Path of json file to save ranks')
parser.add_argument("--log", action="store_true", help="Use weights and biases logging")

# ----------------------------------------------------------------------------
# input arguments and options
# ----------------------------------------------------------------------------

args = parser.parse_args()

if args.log:
    wandb.init(project="a_vs_c", name=args.jobname)
    wandb.config.update(args)

# seed for reproducibility
torch.manual_seed(1234)

# set device and default tensor type
if args.gpuid >= 0:
    torch.cuda.manual_seed_all(1234)
    torch.cuda.set_device(args.gpuid)

# ----------------------------------------------------------------------------
# read saved model and args
# ----------------------------------------------------------------------------
args.load_path = resolve_path(args.load_path)
args.save_path = resolve_path(args.save_path)


components = torch.load(args.load_path)
model_args = components['model_args']
model_args.gpuid = args.gpuid
model_args.batch_size = args.batch_size

# set this because only late fusion encoder is supported yet
args.concat_history = True

for arg in vars(args):
    print('{:<20}: {}'.format(arg, getattr(args, arg)))

# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------

dataset = VisDialDataset(args, [args.split])
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=dataset.collate_fn)

# iterations per epoch
setattr(args, 'iter_per_epoch', math.ceil(dataset.num_data_points[args.split] / args.batch_size))
print("{} iter per epoch.".format(args.iter_per_epoch))

# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------

encoder = Encoder(model_args)
encoder.load_state_dict(components['encoder'])

decoder = Decoder(model_args, encoder)
decoder.load_state_dict(components['decoder'])
print("Loaded model from {}".format(args.load_path))

if args.gpuid >= 0:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

print("Evaluation start time: {}".format(
    datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')))
encoder.eval()
decoder.eval()



if args.use_gt:
    # ------------------------------------------------------------------------
    # calculate automatic metrics and finish
    # ------------------------------------------------------------------------
    all_ranks = []
    for i, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = Variable(batch[key], volatile=True)
                if args.gpuid >= 0:
                    batch[key] = batch[key].cuda()

        enc_out = encoder(batch)
        dec_out = decoder(enc_out, batch)
        ranks = scores_to_ranks(dec_out.data)
        gt_ranks = get_gt_ranks(ranks, batch['ans_ind'].data)
        all_ranks.append(gt_ranks)
    all_ranks = torch.cat(all_ranks, 0)
    process_ranks(all_ranks)
    gc.collect()
else:
    # ------------------------------------------------------------------------
    # prepare json for submission
    # ------------------------------------------------------------------------
    ranks_json = []
    for i, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = Variable(batch[key], volatile=True)
                if args.gpuid >= 0:
                    batch[key] = batch[key].cuda()

        enc_out = encoder(batch)
        dec_out = decoder(enc_out, batch)
        ranks = scores_to_ranks(dec_out.data)
        ranks = ranks.view(-1, 10, 100)

        for i in range(len(batch['img_fnames'])):
            # cast into types explicitly to ensure no errors in schema
            if args.split == 'test':
                ranks_json.append({
                    'image_id': batch['img_fnames'][i][-9:-4],
                    'round_id': int(batch['num_rounds'][i]),
                    'ranks': [rnk.item() for rnk in list(ranks[i][batch['num_rounds'][i] - 1])]
                })
            else:
                for j in range(batch['num_rounds'][i]):
                    ranks_json.append({
                        'image_id': batch['img_fnames'][i][-9:-4],
                        'round_id': int(j + 1),
                        'ranks': [rnk.item() for rnk in list(ranks[i][j])]
                    })
        gc.collect()

if args.save_ranks:
    print("Writing ranks to {}".format(args.save_path))
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    json.dump(ranks_json, open(args.save_path, 'w'))
    if args.log:
        wand.save(args.save_path)
