import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import json
import random

class VIST_img(Dataset):
    def __init__(self, mode, img_path='/home/joe32140/data/VIST/images/all_fixed_rotation_images'):
        super().__init__()
        self.img_path = img_path
        with open(f'./data/{mode}.json') as f:
            self.data = json.load(f)['imgs_seq']
        
        if mode=='train' or mode=='train_video':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        self.y_choices = [[0,1,2,3],
                        [0,2,1,3],
                        [0,2,3,1],
                        [0,1,3,2],
                        [0,3,1,2],
                        [0,3,2,1],
                        [1,0,2,3],
                        [1,0,3,2],
                        [1,2,0,3],
                        [1,3,0,2],
                        [2,0,1,3],
                        [2,1,0,3]]
        self.y_choices_idx = list(range(12))

    def __getitem__(self, idx):
        y = random.sample(self.y_choices_idx, k=1)[0]
        li = self.y_choices[y]
        x = []
        for i in li:
            img = Image.open(os.path.join(self.img_path, f'{self.data[idx][i][0]}.jpg')).convert('RGB')
            img = self.transform(img)
            x.append(img)
        return x, torch.tensor(y)
    
    def __len__(self):
        return len(self.data)