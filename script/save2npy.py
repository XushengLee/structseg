import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt

path = '/mnt/HDD/datasets/competitions/HaN_OAR/'
fs =[os.path.join(path, f) for f in os.listdir(path)]

# for idx,f in enumerate(fs):
#     data = nib.load(os.path.join(f, 'data.nii.gz'))
#     data = data.get_data()
#     data = np.transpose(data, (2,1,0))
#     np.save(os.path.join(f, 'data.npy'), data)
#
#     label = nib.load(os.path.join(f, 'label.nii.gz'))
#     label = label.get_data()
#     label = np.transpose(label, (2, 1, 0))
#     np.save(os.path.join(f, 'label.npy'), label)
#     print(idx)
#
a = np.load(fs[0]+'/data.npy')
print(a.shape, np.unique(a))

#a = np.load('/mnt/HDD/datasets/HaN_OAR/train/12/normed.npy')
plt.imshow(a[70])
plt.show()