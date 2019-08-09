import json
import os
import random
from PIL import Image
import numpy as np
from itertools import combinations

images_folder = '/home/karljackab/2017reshuffling/data/video/imgs'
img_set_number = 4
train_prob = 0.7
test_prob = 0.3

############
## select specific time duration to clip video
## 0.5 second => s = 5
## 1.0 second => s = 10
## 2.0 second => s = 20
s = 20
############

# ========
# result
# {
#     'imgs_seq':
#     [
#       ['file1','file2','file3','file4','file5'],
#       []
#     ]
# }

total_json = {}
imgs_list = os.listdir(images_folder)

for img in imgs_list:
    spl = img.split('_')
    time = round(float(spl[-1][:-4]), 1)
    name = '_'.join(spl[:-1])

    if name not in total_json:
        total_json[name] = []
    total_json[name].append((img[:-4], time))

res = [[], []]  # [train, test]
for video in total_json:
    frames = total_json[video]
    frames.sort(key = lambda x: x[1])

    ## Random Sample
    # frame_len = len(frames)
    # len_list = list(range(frame_len))
    # if frame_len < img_set_number:
    #     continue
    # for _ in range(int(frame_len)*10):
    #     temp = []
    #     idxes = random.sample(len_list, k=img_set_number)
    #     idxes.sort()
    #     for idx in idxes:
    #         temp.append(frames[idx])
    #     mode = np.random.choice([0, 1], 1, p = [train_prob, test_prob])
    #     res[mode[0]].append(temp)

    ## All Combination
    # all_comb = combinations(frames, 4)
    # for comb in all_comb:
    #     temp = []
    #     for i in range(4):
    #         temp.append(comb[i])
    #     mode = np.random.choice([0, 1], 1, p = [train_prob, test_prob])
    #     res[mode[0]].append(temp)

    ## Sample in sequence
    for idx in range(len(frames)-3*s):
        temp = []
        for i in range(idx, idx+3*s+1, s):
            temp.append(frames[i])
        if len(temp) < 4:
            print('length insufficient, aborted')
        mode = np.random.choice([0, 1], 1, p = [train_prob, test_prob])
        res[mode[0]].append(temp)

print(f'train length: {len(res[0])}')
print(f'test length: {len(res[1])}')
with open(f'data/train_{float(s)/10}_video.json', 'w') as f:
    data = {'imgs_seq':res[0]}
    json.dump(data, f)
with open(f'data/test_{float(s)/10}_video.json', 'w') as f:
    data = {'imgs_seq':res[1]}
    json.dump(data, f)