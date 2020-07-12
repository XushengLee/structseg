import os
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T

from torchvision.transforms import functional as F
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import time
from transform.trans import *


class HaN_OAR(data.Dataset):
    def __init__(self, root, fold=1, foreground=False,augmentation_prob=0.5, is_train=True, return_original=False):
        self.root = root
        self.is_train = is_train
        self.fold = fold
        self.cases = [os.path.join(self.root, f) for f in os.listdir(self.root)]
        self.test_cases = [os.path.join(self.root, str(f)) for f in range((fold-1)*10+1,fold*10+1)]
        self.train_cases = [f for f in self.cases if f not in self.test_cases]
        if is_train:
            self.cases = self.train_cases
        else:
            self.cases = self.test_cases
        self.augmentation_prob = augmentation_prob
        self.return_original = return_original
        self.foreground = foreground
        if self.return_original:
            self.is_train, self.augmentation_prob = False, 0.

    def __getitem__(self, index):
        image = np.load(os.path.join(self.cases[index], 'data256.npy'))
        label = np.load(os.path.join(self.cases[index], 'label256.npy'))


        x,y,rx,ry = RandomCrop(train=self.is_train and True if np.random.rand()>self.augmentation_prob else False
                               , spectrum=8)(image)

        if self.is_train and np.random.rand() > self.augmentation_prob:
            image_trans = T.Compose([
                Flip3d(),
                RandomContrast(),
                Crop(x,y,rx,ry),
                CLAHE(),
                FillZ(),
                IntoTensor()
            ])
            if self.foreground:
                label_trans = T.Compose([
                    Flip3d(),
                    Crop(x,y,rx,ry),
                    LabelReduction(),
                    ForegroundLabel(),
                    FillZ(),
                    IntoTensor(type='label')
                ])
            else:
                label_trans = T.Compose([
                    Flip3d(),
                    Crop(x, y, rx, ry),
                    LabelReduction(),
                    FillZ(),
                    IntoTensor(type='label')
                ])

        else:
            image_trans = T.Compose([
                RandomContrast(),
                Crop(x, y, rx, ry),
                CLAHE(),
                FillZ(),
                IntoTensor()
            ])
            if self.foreground:
                label_trans = T.Compose([
                    Crop(x, y, rx, ry),
                    LabelReduction(),
                    ForegroundLabel(),
                    FillZ(),
                    IntoTensor(type='label')
                ])
            else:
                label_trans = T.Compose([
                    Crop(x, y, rx, ry),
                    LabelReduction(),
                    FillZ(),
                    IntoTensor(type='label')
                ])


        if not self.return_original:
            return image_trans(image), label_trans(label)
        else:
            return image_trans(image), label_trans(label), self.cases[index]

    def __len__(self):
        return len(self.cases)




class HaN_OAR_v2(data.Dataset):
    def __init__(self, root, fold=1,is_aug=True, is_train=True, return_params=False,
                 test_clahe=True, test_on = 'raw'):
        self.root = root
        self.is_train = is_train
        self.fold = fold

        self.test_clahe = test_clahe

        self.cases = [str(f) for f in range(1,51)]  # '1' ~ '50'
        self.test_cases = [str(f) for f in range((fold-1)*10+1,fold*10+1)]
        self.train_cases = [f for f in self.cases if f not in self.test_cases]
        if is_train:
            self.cases = self.train_cases
        else:
            self.cases = self.test_cases
        self.is_aug = is_aug
        self.return_params = return_params
        self.test_on = test_on


    def __getitem__(self, index):
        image, label = None, None
        if self.is_train and self.is_aug:
            if np.random.rand() > 0.6:
                image = np.load(
                    os.path.join(self.root, 'raw', self.cases[index],
                                 'data-z128-halved.npy' if np.random.rand()>0.5 else 'data-z128-halved-clahe.npy'))
                label = np.load(os.path.join(self.root, 'raw', self.cases[index], 'label-z128-halved.npy'))

            elif np.random.rand() > 0.3:
                image = np.load(
                    os.path.join(self.root, 'left', self.cases[index],
                                 'data-z128-halved.npy' if np.random.rand()>0.5 else 'data-z128-halved-clahe.npy'))
                label = np.load(os.path.join(self.root, 'left', self.cases[index], 'label-z128-halved.npy'))
            else:
                image = np.load(
                    os.path.join(self.root, 'right', self.cases[index],
                                 'data-z128-halved.npy' if np.random.rand()>0.5 else 'data-z128-halved-clahe.npy'))
                label = np.load(os.path.join(self.root, 'right', self.cases[index], 'label-z128-halved.npy'))

        else:
            image = np.load(
                os.path.join(self.root, self.test_on, self.cases[index],
                             'data-z128-halved-clahe.npy' if self.test_clahe else 'data-z128-halved.npy'))
            label = np.load(os.path.join(self.root, self.test_on, self.cases[index], 'label-z128-halved.npy'))




        x,y,rx,ry = RandomCrop(train=self.is_train and self.is_aug and True if np.random.rand()>0 else False
                               , spectrum=8)(image)

        image_trans, label_trans = [], []

        image_trans.extend([Crop(x,y,rx,ry)])
        label_trans.extend([Crop(x,y,rx,ry), LabelReduction()])

        if self.is_train and self.is_aug:
            # image_trans.append(NumpyNorm())
            if np.random.rand() > 0.5:
                image_trans.extend([NumpyNorm(), RandomContrast(factor=0.05)])
            if np.random.rand() > 0.5:
                image_trans.append(Flip3d())
                label_trans.append(Flip3d())


        image_trans.append(IntoTensor(type='image'))
        label_trans.append(IntoTensor(type='label'))

        if not self.return_params:
            return T.Compose(image_trans)(image), T.Compose(label_trans)(label)
        else:
            return T.Compose(image_trans)(image), T.Compose(label_trans)(label), (x,y,rx,ry), self.cases[index]

    def __len__(self):
        return len(self.cases)






def get_loader(image_path, batch_size ,foreground=True, num_workers=2, is_train=True, augmentation_prob=0.5):
    """Builds and returns Dataloader."""

    dataset = HaN_OAR(image_path, foreground, augmentation_prob, is_train)
    data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader





if __name__ == '__main__':

    set = HaN_OAR_v2('/mnt/HDD/datasets/competitions/aug',is_train=False, is_aug=False,return_params=True,fold=2)
    print(len(set))
    # for ims,gts in set:
    #     print(ims.shape, gts.shape)
    #     plot2d(ims[0,90],bar=True)
    #     plot2d(gts[0,90],bar=True)
    a,_,_,x = set[0]
    b,_,_,_ = set[0]
    plot2d(a[0,90])
    plot2d(b[0,90])
    print(x)