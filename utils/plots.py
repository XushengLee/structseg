import matplotlib.pyplot as plt
import time
import numpy as np
def plot2d(im, bar=False):
    plt.imshow(np.array(im))
    if bar:
        plt.colorbar()
    plt.show()
    time.sleep(0.5)


def plot3d(im):
    pass