# import os
# import skimage.transform as sktrans
# from skimage.exposure import equalize_adapthist
# import numpy as np
# from transform.trans import *
# from utils.plots import plot2d
#
#
# path = '/mnt/HDD/datasets/competitions/aug/raw/3/'
#
# ims = np.load(path+'data-z128.npy')
# gts = np.load(path+'label-z128.npy')
#
# plot2d(ims[70],bar=True)
# plot2d(gts[70],bar=True)
# # plot2d(ims[92],bar=True)
# # plot2d(gts[92],bar=True)
# #
# # print(np.unique(ims[80]))
# # print(np.unique(gts[80]))


from datetime import datetime
from threading import Timer

x=datetime.today()
y=x.replace(day=x.day+1, hour=1, minute=0, second=0, microsecond=0)
delta_t=y-x

secs=delta_t.seconds+1

def hello_world():
    print("hello world")
    #...

t = Timer(secs, hello_world)
t.start()