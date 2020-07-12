import os
import skimage.transform as sktrans
from skimage.exposure import equalize_adapthist
import numpy as np
from transform.trans import *
from utils.plots import plot2d


# the patient's NO: string
# root = '/mnt/HDD/datasets/competitions/HaN_OAR/'

root = '/mnt/HDD/datasets/competitions/aug/raw'

folder = '3'

raw_data = np.load(os.path.join(root, folder, 'data-z128.npy'))
raw_label = np.load(os.path.join(root, folder, 'label-z128.npy'))


# im = raw_data[85]
# gt = raw_label[85]

# plot2d(im,bar=True)
# plot2d(gt,bar=True)

# lim = sktrans.rotate(im,3, preserve_range=True, cval=-1000)
# lgt = sktrans.rotate(gt,3, preserve_range=True )
# rim = sktrans.rotate(im,-3, preserve_range=True, cval=-1000)
# rgt = sktrans.rotate(gt,-3, preserve_range=True )

ims = []
gts = []
for idx in range(128):
    im = raw_data[idx]
    gt = raw_label[idx]
    rtgt = np.zeros((512,512))
    for lb in np.unique(gt):
        if lb != 0:
            t = np.zeros((512,512))
            t[gt == lb] = 1
            rt = sktrans.rotate(t, 3, preserve_range=True)
            rt[rt>=0.5] = 1
            rtgt[rt==1] = lb
    rtgt=rtgt.astype(int)
    print(np.unique(rtgt))
    ims.append(sktrans.rotate(im, 3, preserve_range=True, cval=-1000))
    gts.append(rtgt)

ims = np.array(ims)
gts = np.array(gts)

plot2d(ims[90],bar=True)
plot2d(gts[90],bar=True)