import numpy as np
from utils.plots import plot2d

loaded = np.load('/mnt/HDD/datasets/competitions/set4unet/fold1-right/1-right-90.npz')
plot2d(loaded['clahe'],bar=True)
