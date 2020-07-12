# Contrast Limited Adaptive Histogram Equalization (CLAHE)
# 2D效果很好 但是需要实现3D的情况 不然Z轴上不一致
# update: 即使是2d slice一张一张处理效果仍然很好 可用

from skimage.exposure import equalize_adapthist
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    ims = np.load('/mnt/HDD/datasets/HaN_OAR/train/10/normed.npy')
    ims = (ims - ims.min()) / (ims.max() - ims.min())
    for im in ims:

        t = equalize_adapthist(im, )
        plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(im)
        plt.colorbar()
        plt.subplot(2,1,2)
        plt.imshow(t*im.max())
        plt.colorbar()
        plt.show()
        time.sleep(1)