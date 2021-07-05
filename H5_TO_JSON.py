import os, sys
import h5py
import pickle
from tqdm import tqdm

import numpy as np
import misc.myutils as myutils

if __name__ == "__main__":
    vqa = False
    gqa = True
    if vqa:
        objs_path = "/home/jumperkables/kable_management/data/vqa/features/coco-bottom-up/trainval"
        objs_h5 = h5py.File(os.path.join(objs_path, "features.h5"), 'r')

        import ipdb; ipdb.set_trace()
        for key in tqdm(objs_h5.keys()):
            pickle_save = os.path.join(objs_path, f"{key}.pickle")
            data = objs_h5[key]
            to_save = {
                'bboxes':       np.array(data['bboxes']),
                'features':     np.array(data['features']),
                'image_h':      np.array(data['image_h']),
                'image_w':      np.array(data['image_w']),
                'num_boxes':    np.array(data['num_boxes'])
            }
            myutils.save_pickle(to_save, pickle_save)

    if gqa:
        objs_path = "/mnt/void2/gqa_objects"
        save_path = "/home/jumperkables/kable_management/projects/a_vs_c/data/gqa/objects"
        objects_json = myutils.load_json("/home/jumperkables/kable_management/projects/a_vs_c/data/gqa/objects/gqa_objects_info.json")
        objects_h5s = {
            0:h5py.File(os.path.join(objs_path, "gqa_objects_0.h5"), "r"),#, driver=None),
            1:h5py.File(os.path.join(objs_path, "gqa_objects_1.h5"), "r"),#, driver=None),
            2:h5py.File(os.path.join(objs_path, "gqa_objects_2.h5"), "r"),#, driver=None),
            3:h5py.File(os.path.join(objs_path, "gqa_objects_3.h5"), "r"),#, driver=None),
            4:h5py.File(os.path.join(objs_path, "gqa_objects_4.h5"), "r"),#, driver=None),
            5:h5py.File(os.path.join(objs_path, "gqa_objects_5.h5"), "r"),#, driver=None),
            6:h5py.File(os.path.join(objs_path, "gqa_objects_6.h5"), "r"),#, driver=None),
            7:h5py.File(os.path.join(objs_path, "gqa_objects_7.h5"), "r"),#, driver=None),
            8:h5py.File(os.path.join(objs_path, "gqa_objects_8.h5"), "r"),#, driver=None),
            9:h5py.File(os.path.join(objs_path, "gqa_objects_9.h5"), "r"),#, driver=None),
            10:h5py.File(os.path.join(objs_path, "gqa_objects_10.h5"), "r"),#, driver=None),
            11:h5py.File(os.path.join(objs_path, "gqa_objects_11.h5"), "r"),#, driver=None),
            12:h5py.File(os.path.join(objs_path, "gqa_objects_12.h5"), "r"),#, driver=None),
            13:h5py.File(os.path.join(objs_path, "gqa_objects_13.h5"), "r"),#, driver=None),
            14:h5py.File(os.path.join(objs_path, "gqa_objects_14.h5"), "r"),#, driver=None),
            15:h5py.File(os.path.join(objs_path, "gqa_objects_15.h5"), "r"),#, driver=None)
        }

        #ih5_file, ih5_idx = self.objects_json[img_id]['file'], self.objects_json[img_id]['idx']
        #features = torch.from_numpy(self.objects_h5s[ih5_file]['features'][ih5_idx][:self.n_objs])      
        import ipdb; ipdb.set_trace()
        for img_id in tqdm(objects_json.keys()):
            ih5_file, ih5_idx = objects_json[img_id]['file'], objects_json[img_id]['idx']           
            data = {
                'bboxes':objects_h5s[ih5_file]['bboxes'][ih5_idx],
                'features':objects_h5s[ih5_file]['features'][ih5_idx]
            }
            myutils.save_pickle(data, os.path.join(save_path, f"{img_id}.pickle"))
