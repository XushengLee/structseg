import numpy as np
import os
import matplotlib.pyplot as plt
import time
import SimpleITK as sitk

path = '/mnt/EXTRA/deployments/final-test/'
output_dir = '/mnt/EXTRA/deployments/final/output'

def mirroring_label(label):
    tmp = np.zeros(label.shape)
    tmp[label == 1] =1
    tmp[label == 3] =2
    tmp[label == 2] =3
    tmp[label == 5] =4
    tmp[label == 4] =5
    tmp[label == 7] =6
    tmp[label == 6] =7
    tmp[label == 8] =8
    tmp[label == 10] =9
    tmp[label == 9] =10
    tmp[label == 11] =11
    tmp[label == 13] =12
    tmp[label == 12] =13
    tmp[label == 15] =14
    tmp[label == 14] =15
    tmp[label == 17] =16
    tmp[label == 16] =17
    tmp[label == 19] =18
    tmp[label == 18] =19
    tmp[label == 20] =20
    tmp[label == 22] =21
    tmp[label == 21] =22
    return tmp



for folder in os.listdir(path):
    data = np.load(os.path.join(path, folder, 'data-z128.npy'))
    label = np.load(os.path.join(path, folder, 'label-z128.npy'))

    mirrored_data = data[:,:,::-1]
    mirrored_label = mirroring_label(label[:,:,::-1])
    # print(np.unique(data))
    # plt.imshow(mirrored_data[80])
    # plt.show()
    # time.sleep(2)
    # plt.imshow(mirrored_label[80])
    # plt.show()
    # time.sleep(2)
    # plt.close()
    itk_data = sitk.GetImageFromArray(data.astype(np.int16))
    itk_label = sitk.GetImageFromArray(label.astype(np.int16))

    # return itk_res
    if not os.path.exists(os.path.join(output_dir, folder)):
        os.makedirs(os.path.join(output_dir, folder))
    sitk.WriteImage(itk_data, os.path.join(output_dir, folder, 'data.nii.gz'))
    sitk.WriteImage(itk_label, os.path.join(output_dir, folder, 'label.nii.gz'))