import numpy as np
import sklearn.mixture as mixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time

class Features(object):
    def __init__(self, data, K=2):
        self.data = data
        self.K = K
        self.coords = np.array(list(map(lambda x,y,z: (x,y,z),*np.where(data))))
        if K == 1:
            self.mean = self.coords.mean(axis=0)
            self.cov = np.cov(np.where(self.data))
        else:
            self.gmm = mixture.GaussianMixture(n_components=K)
            self.gmm.fit(self.coords)
            self.mean = self.gmm.means_
            self.cov = self.gmm.covariances_

if __name__ == '__main__':
    data = np.load('/mnt/HDD/datasets/118.npy')
    d= data.copy()

    fs = Features(d,1)
    print(fs.mean, '\n', '\n', fs.cov.shape, fs.cov)
    for label in [2,3,5]:
        d = data.copy()
        d[data != label] = 0

        if label == 5:
            fs = Features(d, 2)
            t1 = np.random.multivariate_normal(fs.mean[0], fs.cov[0], 1000)
            t2 = np.random.multivariate_normal(fs.mean[1], fs.cov[1], 1000)
            print(t1.shape)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(t1[:,0], t1[:,1], t1[:,2])
            ax.scatter(t2[:,0], t2[:,1], t2[:,2], c='r', marker='^')
            ax.set_xlabel('X label')
            ax.set_ylabel('Y label')
            ax.set_zlabel('Z label')
            ax.view_init(elev=30, azim=100)
            plt.show()
        else:
            fs = Features(d,1)
            t = np.random.multivariate_normal(fs.mean, fs.cov, 1000)
            print(t.shape)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(t[:, 0], t[:, 1], t[:, 2])
            ax.set_xlabel('X label')
            ax.set_ylabel('Y label')
            ax.set_zlabel('Z label')
            ax.view_init(elev=30, azim=100)
            plt.show()
        time.sleep(1)