import os, sys
import h5py
from . import myutils
#import myutils
from tqdm import tqdm
#from multimodal.datasets import VQA, VQA2
import csv
import numpy as np
import base64



def frames_to_resnet_h5(dset, h5_save_path):
    import torch
    from torchvision.transforms import ToTensor
    from torchvision.models import resnet152, resnet101, resnet50
    import cv2
    #TODO consider if this kind of importing is worth it
    """
    dset:           GQA, VQACP only
    h5_save_path:   Where to save the h5
    """
    print(f"Processing ResNet information for {dset} dataset...")
    assert dset in ["GQA", "VQACP"], f"Only GQA/VQACP is implemented, you asked for {dset}. You naughty dog."
    if dset == "GQA":
        frames_rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/gqa/images")
        resnet_h5 = h5py.File(h5_save_path, "w", driver=None)
        resnet = resnet152(pretrained=True).cuda()
        resnet.fc = myutils.Identity()
        for param in resnet.parameters():
            param.requires_grad = False
        frames = os.listdir(frames_rootdir)
        for frame in tqdm(frames, total=len(frames)):
            frame_id = frame.split(".")[0]
            # Image feature
            image_path = os.path.join(frames_rootdir, f"{frame_id}.jpg")
            image = torch.from_numpy(cv2.imread(image_path)).permute(2,0,1).cuda() # (channels, height, width)
            height = image.shape[1]
            width = image.shape[2]
            image_feat = resnet(image.float().unsqueeze(0)).squeeze(0)
            image_feat = image_feat.cpu()
            # save height, width, 
            grp = resnet_h5.create_group(frame_id)
            grp.create_dataset('image_h', data=np.asarray([height]) )
            grp.create_dataset('image_w', data=np.asarray([width]) )
            grp.create_dataset('resnet', data=np.asarray(image_feat))

    # VQACP
    if dset == "VQACP":
        resnet_h5 = h5py.File(h5_save_path, "w", driver=None)
        resnet = resnet152(pretrained=True).cuda()
        resnet.fc = myutils.Identity()
        for param in resnet.parameters():
            param.requires_grad = False
        train_frames_rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/images", "train2014")
        frames = os.listdir(train_frames_rootdir)
        for frame in tqdm(frames, total=len(frames)):
            frame_id = int(frame.split("_")[2].split(".")[0])
            # Image feature
            image_path = os.path.join(train_frames_rootdir, frame)
            image = torch.from_numpy(cv2.imread(image_path)).permute(2,0,1).cuda() # (channels, height, width)
            height = image.shape[1]
            width = image.shape[2]
            image_feat = resnet(image.float().unsqueeze(0)).squeeze(0)
            image_feat = image_feat.cpu()
            # save height, width, 
            grp = resnet_h5.create_group(str(frame_id))
            grp.create_dataset('image_h', data=np.asarray([height]) )
            grp.create_dataset('image_w', data=np.asarray([width]) )
            grp.create_dataset('resnet', data=np.asarray(image_feat))        
        valid_frames_rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/vqa/images", "val2014")
        frames = os.listdir(valid_frames_rootdir)
        for frame in tqdm(frames, total=len(frames)):
            frame_id = int(frame.split("_")[2].split(".")[0])
            if str(frame_id) in resnet_h5.keys():
                raise NotImplementedError("Oh no, repeat image_id across val and train split")
            # Image feature
            image_path = os.path.join(valid_frames_rootdir, frame)
            image = torch.from_numpy(cv2.imread(image_path)).permute(2,0,1).cuda() # (channels, height, width)
            height = image.shape[1]
            width = image.shape[2]
            image_feat = resnet(image.float().unsqueeze(0)).squeeze(0)
            image_feat = image_feat.cpu()
            # save height, width, 
            grp = resnet_h5.create_group(str(frame_id))
            grp.create_dataset('image_h', data=np.asarray([height]) )
            grp.create_dataset('image_w', data=np.asarray([width]) )
            grp.create_dataset('resnet', data=np.asarray(image_feat)) 
    resnet_h5.close()


def vqa_tsv_to_h5(tsv_path, h5_dest_path):
    # Features were obtained from: https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip
    # TODO Download instructions
    # This is an adaption made by Jumperkables from code supplied in the 'multimodal' pypip package.
    FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features"]

    names = {}
    with open(tsv_path, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES)
        for item in tqdm(reader):
            # This may seem clunky, but it actually avoids storing the entire object in memory and still runs very fast
            if not os.path.exists(h5_dest_path):
                feats = h5py.File(h5_dest_path, "w", driver=None)
            else:
                feats = h5py.File(h5_dest_path, "a", driver=None)
            item["image_id"] = int(item["image_id"])
            item["image_h"] = int(item["image_h"])
            item["image_w"] = int(item["image_w"])
            item["num_boxes"] = int(item["num_boxes"])
            for field in ["boxes", "features"]:
                item[field] = np.frombuffer(
                    base64.decodebytes(item[field].encode("ascii")),
                    dtype=np.float32,
                ).reshape((item["num_boxes"], -1))
            #names.add(item["image_id"])
            feats.create_group(str(item["image_id"]))
            grp = feats.get(str(item["image_id"]))
            grp.create_dataset('image_h', data=np.asarray([item["image_h"]]) )
            grp.create_dataset('image_w', data=np.asarray([item["image_w"]]) )
            grp.create_dataset('num_boxes', data=np.asarray([item['num_boxes']]) )
            grp.create_dataset('bboxes', data=item["boxes"])
            grp.create_dataset('features', data=item["features"])#, compression="gzip", compression_opts=9)
            feats.close()
            #with outzip.open(str(item["image_id"]), "w") as itemfile:
            #    pickle.dump(item, itemfile)
        #feats.close()



#def download_gqa():
#    pip_mm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/pip_multimodal")
#    import time
#    print("VQA Dataset not Downloaded. Downloading in 5 seconds...")
#    time.sleep(5)
#    DOWNLOAD = VQA(
#        dir_data=pip_mm_path,
#        features="coco-bottomup",
#        split="train",
#        label="best",
#        min_ans_occ=1
#    )
#    DOWNLOAD = VQA(
#        dir_data=pip_mm_path,
#        features="coco-bottomup",
#        split="val",
#        label="best",
#        min_ans_occ=1
#    )
#    DOWNLOAD = VQA(
#        dir_data=pip_mm_path,
#        features="coco-bottomup",
#        split="test",
#        label="best",
#        min_ans_occ=1
#    )
#    # multilabel
#    DOWNLOAD = VQA(
#        dir_data=pip_mm_path,
#        features="coco-bottomup",
#        split="train",
#        label="multilabel",
#        min_ans_occ=1
#    )
#    DOWNLOAD = VQA(
#        dir_data=pip_mm_path,
#        features="coco-bottomup",
#        split="val",
#        label="multilabel",
#        min_ans_occ=1
#    )
#    DOWNLOAD = VQA(
#        dir_data=pip_mm_path,
#        features="coco-bottomup",
#        split="test",
#        label="multilabel",
#        min_ans_occ=1
#    )
#    import sys; sys.exit()




def store_tvqa_cleaned_tokens():
    tvqa_path = os.path.join( os.path.dirname(__file__), "tvqa/tvqa_modality_bias/data")
    tvqa_data = myutils.load_pickle(os.path.join( tvqa_path , "total_dict.pickle" ))
    vcpts = myutils.load_pickle(os.path.join( tvqa_path, "vcpt_features/det_visual_concepts_hq.pickle" ))
    #import ipdb; ipdb.set_trace()
    print(f"Collecting TVQA Questions...")

    # Generate objects for plots
    questions, answers, correct_answers, vcpts_collect, ts_subtitles, nots_subtitles = [], [], [], [], [], []

    for qidx, qdict in tqdm(enumerate(tvqa_data.values()), total=len(tvqa_data)):
        #import ipdb; ipdb.set_trace()
        if len(qdict) == 15:
            a0, a1, a2, a3, a4, answer_idx, q, qid, show_name, ts, vid_name, sub_text, sub_time, located_frame, located_sub_text = qdict.values()
        elif len(qdict) == 14:
            a0, a1, a2, a3, a4, q, qid, show_name, ts, vid_name, sub_text, sub_time, located_frame, located_sub_text = qdict.values()
        else:
            raise ValueError("How did this happen?")
        vcpt = vcpts[vid_name][located_frame[0]:located_frame[1]]
        vcpt = " ".join(list(set([cpt.split()[-1] for line in vcpt for cpt in line.split(" , ") if cpt!="" ])))
        questions.append(q)
        ans = [a0,a1,a2,a3,a4]
        answers += ans
        if len(qdict) == 15:
            correct_answers.append(ans[answer_idx])
        vcpts_collect.append(vcpt)
        ts_subtitles.append(located_sub_text)
        nots_subtitles.append(sub_text)   
    ### Features
    print("cleaning features")
    questions = [ myutils.clean_word(word) for sentence in questions for word in sentence.split() ]
    answers = [ myutils.clean_word(word) for sentence in answers for word in sentence.split() ]
    correct_answers = [ myutils.clean_word(word) for sentence in correct_answers for word in sentence.split() ]
    vcpts_collect = [ myutils.clean_word(word) for sentence in vcpts_collect for word in sentence.split() ]
    ts_subtitles = [ myutils.clean_word(word) for sentence in ts_subtitles for word in sentence.split() ]
    nots_subtitles = [ myutils.clean_word(word) for sentence in nots_subtitles for word in sentence.split() ]
    myutils.save_pickle( {
        "questions":questions,
        "answers":answers,
        "correct_answers":correct_answers,
        "vcpts":vcpts_collect,
        "subs_ts":ts_subtitles,
        "subs_nots":nots_subtitles
    }, os.path.join( os.path.dirname(__file__) , "tvqa/tvqa_modality_bias/data", "cleaned_total_tokens.pickle"))

def load_tvqa_clean_tokens():
    return myutils.load_pickle(os.path.join( os.path.dirname(__file__) , "tvqa/tvqa_modality_bias/data", "cleaned_total_tokens.pickle"))



def load_tvqa_at_norm_threshold(norm="conc-m", norm_threshold=0.7, greater_than=True, include_vid=True, unique_ans=True):
    # Get TVQA datasets
    train_tvqa_dat = myutils.load_json(os.path.abspath(f"{os.path.abspath(__file__)}/../tvqa/tvqa_modality_bias/data/tvqa_train_processed.json"))
    val_tvqa_dat = myutils.load_json(os.path.abspath(f"{os.path.abspath(__file__)}/../tvqa/tvqa_modality_bias/data/tvqa_val_processed.json"))
    if include_vid:
        vid_h5 = h5py.File(os.path.abspath(f"{os.path.abspath(__file__)}/../tvqa/tvqa_modality_bias/data/imagenet_features/tvqa_imagenet_pool5_hq.h5"), "r", driver=None)

    q_n_correcta = [{"q":qdict["q"], "cans":qdict[f"a{qdict['answer_idx']}".lower()], "vid_name":qdict["vid_name"], "located_frame":qdict["located_frame"]} for qdict in train_tvqa_dat+val_tvqa_dat]
    # Get norm dictionary
    norm_dict_path =   os.path.abspath(f"{os.path.abspath(__file__)}/../misc/all_norms.pickle")
    norm_dict = myutils.load_pickle(norm_dict_path)

    # Filter questions with answers that are of certain concreteness
    norm_ans = []
    qs_w_norm_ans = []
    print(f"Collecting TVQA Qs and As of certain concretness")
    for qa in tqdm(q_n_correcta, total=len(q_n_correcta) ):
        q,a  = qa["q"], qa["cans"]  # cans (correct answer)
        try:    # Speedily developing this code, comeback later to replace with .get
            if norm == "conc-m":
                ans_norm = norm_dict.words[a]["conc-m"]["sources"]["MT40k"]["scaled"]
                if greater_than:
                    if ans_norm > norm_threshold:
                        norm_ans.append(a)
                        qs_w_norm_ans.append(qa)
                else:
                    if ans_norm < norm_threshold:
                        norm_ans.append(a)
                        qs_w_norm_ans.append(qa)
        except KeyError:
            pass

    answers =  list(set(norm_ans))
    answers = " ".join(answers)
    if unique_ans:
        norm_seqs = ["@@".join( [qa["q"], answers]) for qa in qs_w_norm_ans]
    else:
        norm_seqs = ["@@".join( [qa["q"], qa["cans"]]) for qa in qs_w_norm_ans]

    if include_vid:  # Load visual features
        norm_imgnt = [ vid_h5[qa["vid_name"]][qa["located_frame"][0]:qa["located_frame"][1]] for qa in qs_w_norm_ans ]
        norm_seqs = (norm_seqs, norm_imgnt)
    return norm_seqs
