import os
from torchvision import utils
import torch
from torch.utils.data import Dataset, IterableDataset
from PIL import Image
import numpy as np
from PIL import Image
import random
import cv2
from torch import nn
import sys
import albumentations as alb
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

import logging

sys.path.append('src/')


class MultiFFDI_phase1(Dataset):
    def __init__(self, phase='train', image_size=224, sample_cnt=10000):

        assert phase in ['train', 'val', 'test']

        image_list, label_list = init_multiFFDI_phase1(phase, only_real=False, sample_cnt=sample_cnt)
        self.image_list = image_list
        self.label_list = label_list
        self.image_size = (image_size, image_size)
        self.phase = phase

        self.transform = self.get_transform()


    def __len__(self):
        return len(self.image_list)
    

    def get_transform(self):
        if self.phase == 'train':
            return alb.Compose([
                alb.RandomResizedCrop(self.image_size[0], self.image_size[1], scale=(0.8, 1.0)),
                alb.HorizontalFlip(),
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=0.3),
                alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
                alb.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
                alb.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.3),
                alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], p=1.0)
        else:
            return alb.Compose([
                alb.Resize(self.image_size[0], self.image_size[1]),
                alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], p=1.0)


    def __getitem__(self, idx):
        filename = self.image_list[idx]
        img = np.array(Image.open(filename))
        img = self.transform(image=img.astype(np.uint8))['image']

        img = img.transpose((2, 0, 1))
        label = self.label_list[idx]

        return img, label


    def collate_fn(self, batch):
        img, label = zip(*batch)
        img = torch.cat([torch.tensor(i).unsqueeze(0) for i in img], dim=0)

        return dict(img=img,
                    label=torch.tensor(label))


    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)





class MultiFFDI_phase1_crop(Dataset):
    def __init__(self, phase='train', image_size=224, sample_cnt=10000):

        assert phase in ['train', 'val', 'test']

        image_list, label_list = init_multiFFDI_phase1(phase, only_real=False, sample_cnt=sample_cnt)

        self.image_list = image_list
        self.label_list = label_list
        self.image_size = (image_size, image_size)
        self.phase = phase

        self.transform = self.get_transform()


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        filename = self.image_list[idx]
        img = np.array(Image.open(filename))
        
        landmark_path = filename.replace('/frames/', '/landmarks/').replace('.jpg', '.npy')
        if os.path.exists(landmark_path):
            landmark = np.load(landmark_path)[0]
            bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
            x0 = int(bbox_lm[0])
            y0 = int(bbox_lm[1])
            x1 = int(bbox_lm[2])
            y1 = int(bbox_lm[3])
            bbox = np.array([[x0, y0], [x1, y1]])

            landmark = self.reorder_landmark(landmark)

            img, landmark, bbox, _ = crop_face(img, landmark, bbox, margin=True, crop_by_bbox=False)
        else:
            landmark = None
        
        img = self.transform(image=img.astype(np.uint8))['image']
        img = img.transpose((2, 0, 1))

        label = self.label_list[idx]

        return img, label


    def reorder_landmark(self,landmark):
        landmark_add=np.zeros((13,2))
        for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
            landmark_add[idx]=landmark[idx_l]
        landmark[68:]=landmark_add
        return landmark


    def collate_fn(self, batch):
        img, label = zip(*batch)
        img = torch.cat([torch.tensor(i).unsqueeze(0) for i in img], dim=0)
        return dict(img=img, label=torch.tensor(label))
    

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    
    def get_transform(self):
        if self.phase == 'train':
            return alb.Compose([
                alb.RandomResizedCrop(self.image_size[0], self.image_size[1], scale=(0.8, 1.0)),
                alb.HorizontalFlip(),
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=0.3),
                alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
                alb.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
                alb.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.3),
                alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], p=1.0)
        else:
            return alb.Compose([
                alb.Resize(self.image_size[0], self.image_size[1]),
                alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], p=1.0)



if __name__ == '__main__':
    from utils.datasets.initialize import *
    from utils.datasets.funcs import crop_face

    seed=10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    image_dataset = MultiFFDI_phase1_crop(phase='val',image_size=256)
    batch_size=64
    dataloader = torch.utils.data.DataLoader(image_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=image_dataset.collate_fn,
                    num_workers=0,
                    worker_init_fn=image_dataset.worker_init_fn
                    )
    data_iter=iter(dataloader)
    data=next(data_iter)
    img=data['img']
    img=img.view((-1,3,256,256))
    utils.save_image(img, 'loader.jpg', nrow=batch_size, normalize=False, value_range=(0, 1))
else:
    from utils import blend as B
    from .initialize import *
    from .funcs import crop_face
