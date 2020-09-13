import json
##
# NOTE: train_options.json i found was ALREADY IN THE RIGHT FORMAT. This script was not needed
##
input_file = '../../../data/AVSD/qa_and_options/train_options.json'
output_file= '../../../data/AVSD/qa_and_options/train_options_2.json'
import ipdb; ipdb.set_trace()
d = json.load(open(input_file, 'r'))
dialogs = d['data']['dialogs']

new_dialogs = []
for video_id in dialogs:
    new_dialogs.append({'image_id': video_id,
                        'caption': dialogs[video_id]['caption'],
                        'dialog':dialogs[video_id]['dialog']})

new_d = {'data': {'questions': d['data']['questions'],
                  'answers': d['data']['answers'],
                  'dialogs': new_dialogs},
         'split': d['split'],
         'version': d['version']}

json.dump(new_d, open(output_file, 'w'))
