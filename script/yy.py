import os
import skimage.transform as sktrans
from skimage.exposure import equalize_adapthist
import numpy as np
from transform.trans import *

# root = '/mnt/HDD/datasets/competitions/aug/raw'
# folders = os.listdir(root)
# for folder in folders:
#     data512 = np.load(os.path.join(root, folder, 'data-z128.npy'))
#     data256 = np.load(os.path.join(root, folder, 'data-z128-halved.npy'))
#
#     # plot2d(data512[70], bar=True)
#     # plot2d(data256[70], bar=True)
#
#     data512 = CLAHE(kernel=32)(data512)
#
#     data256 = CLAHE(kernel=16)(data256)
#
#
#     np.save(os.path.join(root, folder, 'data-z128-clahe.npy'), data512)
#     np.save(os.path.join(root, folder, 'data-z128-halved-clahe.npy'), data256)
#     print(folder)
#


path = '/mnt/HDD/datasets/competitions/aug/right/1'

data512 = np.load(os.path.join(path, 'data-z128-clahe.npy'))
data256 = np.load(os.path.join(path, 'data-z128-halved-clahe.npy'))

plot2d(data512[10],bar=True)
plot2d(data256[10],bar=True)

data512 = CLAHE(kernel=32)(data512)

data256 = CLAHE(kernel=16)(data256)

plot2d(data512[10],bar=True)
plot2d(data256[10],bar=True)