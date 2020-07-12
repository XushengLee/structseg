import numpy as np
import os


ROOTROOT = '/mnt/EXTRA/datasets/competitions'
ROOT = os.path.join(ROOTROOT,'aug')
# specify which fold
FOLD = 5
AUG = 'right'
save2dir = os.path.join(ROOTROOT, 'set4unet')

def set2dify(root, fold, aug, save2dir):
    path = os.path.join(root, aug)
    cases = os.listdir(path)
    for case in cases:
        raw_data = np.load(os.path.join(path, case, 'data-z128.npy'))
        clahe_data = np.load(os.path.join(path, case, 'data-z128-clahe.npy'))
        # (14, 128, 128, 128)
        vnet_output = np.load(os.path.join(path, case, 'vnet-fold{}-z128-halved-clahe.npy'.format(fold)))
        label = np.load(os.path.join(path, case, 'label-z128.npy'))

        for idx in range(128):
            np.savez_compressed(os.path.join(save2dir,'fold{}-{}'.format(fold,aug),'{}-{}-{}.npz'.format(case, aug, idx)),
                                img=raw_data[idx], clahe=clahe_data[idx],
                                vnet=vnet_output[:,idx,:,:], label=label[idx])
        print(case)

set2dify(ROOT, FOLD, 'right', save2dir)
set2dify(ROOT, FOLD, 'left', save2dir)
set2dify(ROOT, FOLD, 'raw', save2dir)

#
# data-z128-clahe.npy           vnet-output-z128-halved-clahe.npy
#   data-z128.npy         label-z128.npy