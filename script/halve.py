import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as ski_transform
import torch

path = '/mnt/HDD/datasets/competitions/aug/right/'
fs =[os.path.join(path, f) for f in os.listdir(path)]

for idx, f in enumerate(fs):



    data = np.load(f+'/data-z128.npy')
    label = np.load(f+'/label-z128.npy')

    data_list = [ski_transform.resize(t, output_shape=(data.shape[1]//2, data.shape[2]//2),
                                               preserve_range=True, order=3) for t in data]
    data = np.array(data_list)

    label_list = []
    for i in range(label.shape[0]):
        wait_label = np.zeros((label.shape[1]//2,label.shape[2]//2))
        for v in range(1,23):
            mask = label[i] == v
            t_label = np.zeros((label.shape[1],label.shape[2]))
            t_label[mask]=1
            t_label = ski_transform.resize(t_label, output_shape=(label.shape[1]//2, label.shape[2]//2),
                                                   preserve_range=True, order=3)
            t_label = t_label.round()*v
            wait_label[t_label==v] = v
        label_list.append(wait_label)

    label = np.array(label_list)

    np.save(f+'/data-z128-halved.npy',data)
    np.save(f+'/label-z128-halved.npy',label)
    print(idx)





# label_list = [ski_transform.resize(t, output_shape=(label.shape[1]//2, label.shape[2]//2),
#                                            preserve_range=True, order=3) for t in label]
# print(np.unique(np.array(label_list)).round())
# plt.imshow(label_list[70])
# plt.show()


