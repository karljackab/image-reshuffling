############################################
## Author: Chih-Chia Li
############################################
## Partially implementation of paper
##   "Unsupervised Representation Learning by Sorting Sequences" (2017ICCV)
############################################

import matplotlib as mpl
mpl.use('Agg')

import os
import matplotlib.pyplot as plt

pths = ['0.5video', '1.0video', '2.0video']
names = ['0.5s', '1.0s', '2.0s']
files = ['train_record','test_record']
basedir = '/home/karljackab/2017reshuffling/'


for pth, name in zip(pths, names):
    pth1 = os.path.join(basedir, pth, files[0])
    pth2 = os.path.join(basedir, pth, files[1])
    with open(pth1) as f:
        temp = f.readlines()
        train = []
        for i in temp:
            train.append(float(i.strip()))
    with open(pth2) as f:
        temp = f.readlines()
        test = []
        for i in temp:
            test.append(float(i.strip()))

    train_max_val = max(train)
    train_max_idx = train.index(train_max_val)
    test_max_val = max(test)
    test_max_idx = test.index(test_max_val)
    plt.text(train_max_idx, train_max_val, f'{train_max_val:.2f}')
    plt.text(test_max_idx, test_max_val, f'{test_max_val:.2f}')

    plt.plot(train, label=f'{name} train')
    plt.plot(test, label=f'{name} test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Video Reshuffling')
plt.savefig('video_result.png')
plt.close()