############################################
## Author: Chih-Chia Li
############################################
## Partially implementation of paper
##   "Unsupervised Representation Learning by Sorting Sequences" (2017ICCV)
############################################

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model.vgg as vgg
import model.dataset as dataset
import model.shufModel as main_model

device = torch.device('cuda')

############
## select specific dataset
## could be "0.5", "1.0", "2.0"
ds = "2.0"
############
train_loader = DataLoader(dataset=dataset.VIST_img(f'train_{ds}_video', img_path='data/video/imgs'),
    batch_size=8,
    shuffle=True,
    num_workers=2)
test_loader = DataLoader(dataset=dataset.VIST_img(f'test_{ds}_video', img_path='data/video/imgs'),
    batch_size=8,
    num_workers=2)

LR = 0.001
Epoch = 50
show_iter = 100
save_dir = f'/home/karljackab/2017reshuffling/{ds}video'

TRAIN = True

model = main_model.shufModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

Loss = nn.CrossEntropyLoss()

test_acc_max = -1
print(f'train length {len(train_loader)}')
print(f'test length {len(test_loader)}')
for epoch in range(Epoch):
    train_acc = 0
    train_tot = 0
    test_acc = 0
    test_tot = 0

    acc_temp, tot_temp = 0, 0
    loss_temp = 0
    tot_loss = 0
    if TRAIN:
        for idx, (x_, y_idx) in enumerate(train_loader):
            x = [ i.to(device) for i in x_ ]
            y_idx = y_idx.to(device)
            pred = model(x)
            loss = Loss(pred, y_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = pred.detach().argmax(1)
            acc = (pred==y_idx).sum()
            tot_temp += len(pred)
            acc_temp += acc
            loss_temp += loss.detach()

            if idx % show_iter == 0:
                train_acc += acc_temp
                train_tot += tot_temp
                tot_loss += loss_temp
                print(f'Epoch {epoch} iter {idx}: {float(acc_temp)/float(tot_temp)}, loss {float(loss_temp)/float(tot_temp)}')
                acc_temp, tot_temp = 0, 0
                loss_temp = 0

        train = float(train_acc)/float(train_tot)
        tot_loss = float(tot_loss)/float(train_tot)

        with open(os.path.join(save_dir, 'train_record'), 'a') as f:
            f.write(str(float(train))+'\n')
        with open(os.path.join(save_dir, 'train_loss'), 'a') as f:
            f.write(str(float(tot_loss))+'\n')

    with torch.no_grad():
        for idx, (x_, y_idx) in enumerate(test_loader):
            x = [ i.to(device) for i in x_ ]
            y_idx = y_idx.to(device)
            pred = model(x)
            y_idx = y_idx.to(device)
            
            pred = pred.detach().argmax(1)
            acc = (pred==y_idx).sum()
            test_tot += len(pred)
            test_acc += acc

            if idx % show_iter == 0:
                print(f'Epoch {epoch} iter {idx}: acc {float(test_acc)/float(test_tot)}')
    
    accuracy = float(test_acc)/float(test_tot)

    if accuracy > test_acc_max:
        test_acc_max = accuracy
        torch.save(model.state_dict(), os.path.join(save_dir, f'{epoch}_{accuracy}.pkl'))

    with open(os.path.join(save_dir, 'test_record'), 'a') as f:
        f.write(str(float(accuracy))+'\n')