import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as ski_transform
import torch
from utils.plots import plot2d

def halve(data, label):
    data_list = [ski_transform.resize(t, output_shape=(data.shape[1]//2, data.shape[2]//2),
                                               preserve_range=True, order=3) for t in data]
    data = np.array(data_list)
    #print(np.unique(label))
    label_list = []
    #print(len(label))
    for idx in range(len(label)):
        gt = label[idx]
        halved = np.zeros((256,256))
        for lb in np.unique(gt):
            if lb != 0:
                t = np.zeros((512,512))
                t[gt==lb] = 1
                t = ski_transform.resize(t, output_shape=(256,256),preserve_range=True, order=3)
                t[t>=0.5] = 1
                halved[t==1] = lb
        halved = halved.astype(int)
        label_list.append(halved)

    label = np.array(label_list)

    return data, label


root = '/mnt/HDD/datasets/competitions/aug/raw/'

folders = os.listdir(root)

for folder in folders:
    data = np.load(os.path.join(root, folder, 'data-z128.npy'))
    label = np.load(os.path.join(root, folder, 'label-z128.npy'))

    halved_data,halved_label = halve(data, label)
    if not os.path.exists(os.path.join(root, folder)):
        os.makedirs(os.path.join(root, folder))

    np.save(os.path.join(root, folder, 'data-z128-halved.npy'), halved_data)
    np.save(os.path.join(root, folder, 'label-z128-halved.npy'), halved_label)
    print(folder)



