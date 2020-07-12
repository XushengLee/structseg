from set.ds3d import HaN_OAR_v2
import numpy as np

aug = 'right'

set1 = HaN_OAR_v2('/mnt/HDD/datasets/competitions/aug',
                  fold=1,is_train=True , is_aug=False,
                  return_params=True, test_on=aug)

set2 = HaN_OAR_v2('/mnt/HDD/datasets/competitions/aug',
                  fold=1,is_train=False , is_aug=False,
                  return_params=True, test_on=aug)

coords = {}
for case in set2:
    _,_,co,name = case
    coords[name]=co
    print(co, name)

for case in set1:
    _,_,co,name = case
    coords[name]=co
    print(co, name)


np.savez('/mnt/HDD/datasets/competitions/set4unet/coords/{}.npz'.format(aug),
         **coords)

print(np.load('/mnt/HDD/datasets/competitions/set4unet/coords/{}.npz'.format(aug))['1'])