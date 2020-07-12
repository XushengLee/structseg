import torch.utils.data as data
import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import pydicom
import numpy as np
import re
from PIL import Image
import nibabel as nib
from torchvision import transforms
from torch.utils.data import DataLoader

from transform.trans import *

class UnetSet(data.Dataset):
    def __init__(self, path, fold=1, is_train=True):

        self.path = path

        self.fold =fold
        self.is_train = is_train

        self.folders = os.listdir(path)

        self.test_folders = [str(f) for f in range((fold - 1) * 10 + 1, fold * 10 + 1)]
        self.train_folders = [f for f in self.folders if f not in self.test_folders]

        if is_train:
            self.folders = self.train_folders
        else:
            self.folders = self.test_folders

        self.probs_root = '/mnt/HDD/datasets/competitions/vnet/output/fold{}'.format(self.fold)

        #dirs = [os.path.join(path, d) for d in os.listdir(path)]

        self.im = []
        self.gt = []
        self.prob = []

        for folder in self.folders:
            img3d = np.load(os.path.join(self.path, folder, 'data256.npy'))
            gt3d = np.load(os.path.join(self.path, folder , 'label256.npy'))
            prob3d = np.load(os.path.join(self.probs_root, 'output{}.npy'.format(folder)))
            # some preprocs for img3d&gt3d, make z always 128
            x, y, rx, ry = RandomCrop(train=False, spectrum=8)(img3d)
            img3d = Crop(x, y, rx, ry)(img3d)
            gt3d = Crop(x, y, rx, ry)(gt3d)
            img3d = FillZ(shape2d=(128,128))(img3d)
            gt3d = FillZ(shape2d=(128,128))(gt3d)
            # some procs for the prob3d, (c,d,h,w)->(d,c,h,w),
            prob3d = np.transpose(prob3d,(1,0,2,3))




            for idx in range(len(img3d)):
                self.im.append(img3d[idx])
                self.gt.append(gt3d[idx])
                self.prob.append(prob3d[idx])

    def __getitem__(self, index):
        img, prob, gt = self.im[index], self.prob[index], self.gt[index]
        img_trans = T.Compose([
            IntoTensor(type='image'),
        ])
        prob_trans = T.Compose([
            #Resize(is_label=False, shape=(14,256,256)),
            IntoTensor(type='image'),
        ])
        gt_trans = T.Compose([
            #LabelReduction(),
            IntoTensor(type='label'),
        ])


        #img, prob, gt = IntoTensor(type='image')(img), IntoTensor(type='label')(prob), IntoTensor(type='label')(gt)

        #return torch.cat((img,prob), dim=0), gt.squeeze()
        return img_trans(img) ,prob_trans(prob).squeeze(), gt_trans(gt)

    def __len__(self):
        return len(self.im)




class UnetSet_v2(data.Dataset):
    def __init__(self, path, fold=1, is_train=True, is_aug=True):

        self.path = path

        self.fold =fold
        self.is_train = is_train
        self.is_aug = is_aug
        self.folders = os.listdir(path)

        self.test_folders = [str(f) for f in range((fold - 1) * 10 + 1, fold * 10 + 1)]
        self.train_folders = [f for f in self.folders if f not in self.test_folders]

        if is_train:
            self.folders = self.train_folders
        else:
            self.folders = self.test_folders


        #dirs = [os.path.join(path, d) for d in os.listdir(path)]

        self.im = []
        self.gt = []
        self.prob = []

        for folder in self.folders:
            img3d = np.load(os.path.join(self.path, folder, 'data-z128-clahe.npy'))
            gt3d = np.load(os.path.join(self.path, folder , 'label-z128.npy'))  # (128,512,512)
            prob3d = np.load(os.path.join(self.path, folder, 'vnet-output-z128-halved-clahe.npy')) #(14,128,128,128)
            # some preprocs for img3d&gt3d, make z always 128
            x, y, rx, ry = RandomCrop(train=False, spectrum=8)(img3d)
            img3d = Crop256(x, y, rx, ry)(img3d)  # (128,256,256)
            gt3d = Crop256(x, y, rx, ry)(gt3d)
            # some procs for the prob3d, (c,d,h,w)->(d,c,h,w),
            prob3d = np.transpose(prob3d,(1,0,2,3))  # (128,114,128,128)




            #for idx in range(len(img3d)):
            self.im.extend(img3d[:])
            self.gt.extend(gt3d[:])
            self.prob.extend(prob3d[:])

    def __getitem__(self, index):
        img, prob, gt = self.im[index], self.prob[index], self.gt[index]
        img_trans, prob_trans, gt_trans = [],[],[]

        if self.is_train and self.is_aug:
            if np.random.rand() > 0.5:
                img_trans, prob_trans = img_trans.append(RandomContrast(0.5)), prob_trans.append(RandomContrast(1))
            if np.random.rand() > 0.5:
                img_trans, prob_trans, gt_trans = img_trans.append(Flip2d()), prob_trans.append(Flip2d()), gt_trans.append(Flip2d())

        img_trans.append(IntoTensor(type='image'))
        prob_trans.extend([Resize(is_label=False,shape=(14,256,256)) ,IntoTensor(type='image')])
        gt_trans.append(IntoTensor(type='label'))



        #img, prob, gt = IntoTensor(type='image')(img), IntoTensor(type='label')(prob), IntoTensor(type='label')(gt)

        #return torch.cat((img,prob), dim=0), gt.squeeze()
        return T.Compose(img_trans)(img) ,T.Compose(prob_trans)(prob), T.Compose(gt_trans)(gt)

    def __len__(self):
        return len(self.im)




class UnetSet_v3(data.Dataset):
    def __init__(self, root, fold=1, is_train=True, is_aug=True):

        self.root = root

        self.fold =fold
        self.is_train = is_train
        self.is_aug = is_aug

        self.cases = list(range(1,51))

        self.test_cases = [str(f) for f in range((fold - 1) * 10 + 1, fold * 10 + 1)]
        self.train_cases = [f for f in self.cases if f not in self.test_cases]

        if is_train:
            self.cases = self.train_cases
        else:
            self.cases = self.test_cases

        if self.is_train and is_aug:
            self.filenames = os.listdir(os.path.join(self.root, 'fold{}-{}'.format(self.fold, 'raw')))\
                             +os.listdir(os.path.join(self.root, 'fold{}-{}'.format(self.fold, 'left')))\
                             +os.listdir(os.path.join(self.root, 'fold{}-{}'.format(self.fold, 'right')))

            self.filenames = [f for f in self.filenames if f.split('-')[0] not in self.test_cases]
        else:
            self.filenames = os.listdir(os.path.join(self.root, 'fold{}-{}'.format(self.fold, 'raw')))
            self.filenames = [f for f in self.filenames if f.split('-')[0] in self.test_cases]




        # cases 1-50, slices 0-127



    def __getitem__(self, index):
        filename = self.filenames[index]

        path = os.path.join(self.root, 'fold{}-{}'.format(self.fold, filename.split('-')[1]),
                            filename)

        loaded = np.load(path)

        img, prob, label = loaded['clahe'] if np.random.rand()>0.5 else loaded['img'], loaded['vnet'], loaded['label']

        img_trans, prob_trans, gt_trans = [], [], []

        if self.is_train and self.is_aug:
            if np.random.rand() > 0.5:
                img_trans.extend([NumpyNorm(), RandomContrast(factor=0.05)])
            if np.random.rand() > 0.5:
                prob_trans.extend([NumpyNorm(), RandomContrast(factor=0.05)])


        x,y,_,_ = np.load(os.path.join(
            self.root, 'coords', '{}.npz'.format(filename.split('-')[1])
        ))[filename.split('-')[0]]
        # print(x,y, img.shape, prob.shape, label.shape)

        rand_rx = np.random.randint(2)
        rand_ry = np.random.randint(2)

        img_trans.append(Crop2d(2*x+rand_rx,2*y+rand_ry,0,0))
        gt_trans.append(Crop2d(2*x+rand_rx,2*y+rand_ry,0,0))

        img_trans.append(IntoTensor(type='image'))
        prob_trans.append(IntoTensor(type='image'))
        gt_trans.append(IntoTensor(type='label'))




        img, prob, label = T.Compose(img_trans)(img), T.Compose(prob_trans)(prob), T.Compose(gt_trans)(label)

        return img, prob.squeeze(), label


    def __len__(self):
        return len(self.filenames)





if __name__ == '__main__':

    set = UnetSet_v3('/mnt/HDD/datasets/competitions/set4unet',is_train=True)
    print(len(set))
    img, prob, gt = set[90]
    print(gt.shape)
    print(img.shape, prob.shape, gt.shape)
    plot2d(img.squeeze().numpy(), bar=True)
    plot2d(gt.squeeze().numpy(), bar=True)
    plot2d(prob.squeeze().numpy()[13], bar=True)
    print(np.unique(gt))

