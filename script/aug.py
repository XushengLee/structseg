import os
import skimage.transform as sktrans
from skimage.exposure import equalize_adapthist
import numpy as np
from transform.trans import *



def rot(raw_data, raw_label, angle):
    ims = []
    gts = []
    for idx in range(len(raw_data)):
        im = raw_data[idx]
        gt = raw_label[idx]
        rtgt = np.zeros((512, 512))
        for lb in np.unique(gt):
            if lb != 0:
                t = np.zeros((512, 512))
                t[gt == lb] = 1
                rt = sktrans.rotate(t, angle, preserve_range=True)
                rt[rt >= 0.5] = 1
                rtgt[rt == 1] = lb
        rtgt = rtgt.astype(int)
        #print(np.unique(rtgt))
        ims.append(sktrans.rotate(im, angle, preserve_range=True, cval=-1000))
        gts.append(rtgt)

    ims = np.array(ims)
    gts = np.array(gts)
    return ims, gts

# the patient's NO: string
# root = '/mnt/HDD/datasets/competitions/HaN_OAR/'

root = '/mnt/HDD/datasets/competitions/aug/raw'
# save2root = '/mnt/HDD/datasets/competitions/aug/final_test/'


# root = '/mnt/EXTRA/deployments/inference-test1/input/'
save2root = '/mnt/EXTRA/deployments/final-test/'

folders = os.listdir(root)

angle = 5

for folder in folders:
    if folder in [str(i) for i in range(1,10)]:
        raw_data = np.load(os.path.join(root, folder, 'data-z128.npy'))
        raw_label = np.load(os.path.join(root, folder, 'label-z128.npy'))

        rot_data, rot_label = rot(raw_data, raw_label, angle)
        if not os.path.exists(os.path.join(save2root, folder)):
            os.makedirs(os.path.join(save2root, folder))

        np.save(os.path.join(save2root, folder, 'data-z128.npy'), rot_data)
        np.save(os.path.join(save2root, folder, 'label-z128.npy'), rot_label)
        print(folder)





