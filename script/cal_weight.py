import os
import skimage.transform as sktrans
from skimage.exposure import equalize_adapthist
import numpy as np
from transform.trans import *
from set.ds3d import HaN_OAR_v2


root = '/mnt/HDD/datasets/competitions/aug/'

set = HaN_OAR_v2(root, is_train=True)

count = np.zeros((14,))
for _, gts in set:
    _,cnt = np.unique(gts, return_counts=True)
    count+=cnt
    print(cnt)
count = count/len(set)

print(count)
#
# [2.06655173e+06 1.78807500e+03 1.27810000e+03 3.54000000e+01
#  9.90000000e+01 6.34750000e+01 1.43936250e+04 3.51250000e+01
#  3.68237500e+03 3.75125000e+02 1.70517500e+03 4.09100000e+02
#  1.48525000e+03 5.25045000e+03]

# a = np.array([2.06655173e+06, 1.78807500e+03, 1.27810000e+03, 3.54000000e+01,
#  9.90000000e+01, 6.34750000e+01, 1.43936250e+04, 3.51250000e+01,
#  3.68237500e+03, 3.75125000e+02, 1.70517500e+03, 4.09100000e+02,
#  1.48525000e+03, 5.25045000e+03])
# print(1/(a/2.06655173e+06))
# [1.00000000e+00 1.15574108e+03 1.61689362e+03 5.83771675e+04
#  2.08742599e+04 3.25569394e+04 1.43574098e+02 5.88342130e+04
#  5.61200782e+02 5.50896829e+03 1.21192941e+03 5.05145864e+03
#  1.39138309e+03 3.93595164e+02]
# [1, 1000, 1500,50000,20000,30000,150,50000,500,5000,1000,5000,1000,400]
# **2/3
  # [1 ,100, 130, 1000, 700, 900, 30, 1000, 60, 200,100,300,100,55]


# weight = torch.tensor(
#                 [1., 100., 100., 50., 80., 50., 80., 80., 50., 70., 70.,
#                  60., 100., 100., ]).to(self.device)