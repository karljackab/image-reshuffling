import torch
import torch.nn as nn
import model.vgg as vgg

class shufModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg.vgg16_bn()
        self.pairwise = nn.Conv2d(1024, 3, kernel_size=(3,3))
        self.order_predict = nn.Linear(5*5*3*6, 12)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        feature_list = []
        for img in x:
            feat = self.vgg(img)
            feature_list.append(feat)
        pairwise_list = []
        for i in range(4):
            for j in range(i+1, 4):
                connect = torch.cat((feature_list[i], feature_list[j]), 1)
                pairs = self.pairwise(connect)
                pairs = self.drop(pairs)
                pairwise_list.append(self.relu(pairs).view(pairs.shape[0], -1))
        
        res = pairwise_list[0]
        for i in range(1, len(pairwise_list)):
            res = torch.cat((res, pairwise_list[i]), 1)
        res = self.order_predict(res)
        return res