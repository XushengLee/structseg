import numpy as np
from utils.plots import plot2d
import scipy.ndimage as ndimage
import skimage.transform as ski_transform
import torch
from skimage.exposure import equalize_adapthist


class Flip2d(object):
    """
    flip the horizontal axis of either (H,W) or (C,H,W)
    """
    def __init__(self):
        self.axis = -1

    def __call__(self, m):
        assert len(m.shape) in [2,3]

        m = np.flip(m,self.axis)

        return np.copy(m)



class Flip3d(object):
    """
    flip the horizontal axis of either (D,H,W) or (C,D,H,W)
    """
    def __init__(self):
        self.axis = -1

    def __call__(self, m):
        assert len(m.shape) in [3,4]

        m = np.flip(m,self.axis)

        return np.copy(m)


class RandomContrast(object):
    """
    jittering on the 3d-image, normed, value ranges from -10 ~ 120
    """
    def __init__(self, factor=8, **kwargs):
        self.factor = factor

    def __call__(self, m):
        r = np.random.randn(*m.shape)
        return m+self.factor*r


class RandomRotation(object):
    """
    will mess up the label, even after rounded.
    """
    def __init__(self, angle_spectrum=10,axes=None,mode='constant', order=0, is_label=False, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order
        self.is_label = is_label

    def __call__(self, m):
        axis = self.axes[np.random.randint(len(self.axes))]
        angle = np.random.randint(-self.angle_spectrum, self.angle_spectrum)
        if len(m.shape) == 3:
            m = ndimage.rotate(m, angle, axes=axis)
        else:
            channels = [ndimage.rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        if self.is_label:
            return m.round()

        return m




class Resize(object):
    """

    """
    def __init__(self, is_label=False, shape=(14,256,256)):
        self.shape = shape
        self.is_label = is_label

    def __call__(self, m):
        m = ski_transform.resize(m, self.shape, preserve_range=True)
        if self.is_label:
            return m.round()
        return m


class FillZ(object):
    def __init__(self, shape2d = (128,128)):
        self.shape2d = shape2d
    def __call__(self, m):
        l = m.shape[0]
        if l >= 128:
            return m[:128]
        else:
            t = np.zeros((128,self.shape2d[0],self.shape2d[1]),dtype=float)
            t[:l] = m
            t[l:] = m[l-1]
            return t



class RandomCrop(object):
    def __init__(self, offset=32, spectrum=20, train=False):
        self.offset = offset
        self.spectrum = spectrum
        self.train = train


    def find_center(self, data):
        threshold = (data.max() - data.min())*0.2+data.min()
        press = np.sum(data,axis=0)/len(data)
        cos = np.where(press > threshold)
        x, y = np.int(cos[0].mean()), np.int(cos[1].mean())
        return x, y

    def __call__(self, m):
        x,y = self.find_center(m)
        rx,ry = 0,0
        if self.train:
            rx = np.random.randint(-self.spectrum, self.spectrum)
            ry = np.random.randint(-self.spectrum, self.spectrum)

        return x,y,rx,ry
    #
    # m[:128,
    # x - 128 - self.offset - rx: x + 128 - self.offset - rx,
    # y - 128 - ry: y + 128 - ry]


class Crop(object):
    def __init__(self,x,y,rx,ry):
        self.x = x
        self.y = y
        self.rx = rx
        self.ry = ry
        self.offset = 16

    def __call__(self, m):
        # if m.shape[0]<128:
        #     dummy = [m[-1] for i in range(128-m.shape[0])]
        #     dummy = np.array(dummy)
        #     m = np.concatenate((m, dummy), axis=0)

        """
        use the method of add 2d-image to make certain chunk of image stack
        """
        # return m[:128,
        #        self.x - 32 - self.offset - self.rx: self.x + 32 - self.offset - self.rx,
        #        self.y - 32 - self.ry: self.y + 32 - self.ry]

        """
        just return the color channel (slices), rely on the Resize to do the job
        """
        return m[:,
               self.x - 64 - self.offset - self.rx: self.x + 64 - self.offset - self.rx,
               self.y - 64 - self.ry: self.y + 64 - self.ry]



class Crop2d(object):
    def __init__(self,x,y,rx,ry):
        self.x = x
        self.y = y
        self.rx = rx
        self.ry = ry
        self.offset = 32

    def __call__(self, m):
        # if m.shape[0]<128:
        #     dummy = [m[-1] for i in range(128-m.shape[0])]
        #     dummy = np.array(dummy)
        #     m = np.concatenate((m, dummy), axis=0)

        """
        use the method of add 2d-image to make certain chunk of image stack
        """
        # return m[:128,
        #        self.x - 32 - self.offset - self.rx: self.x + 32 - self.offset - self.rx,
        #        self.y - 32 - self.ry: self.y + 32 - self.ry]

        """
        just return the color channel (slices), rely on the Resize to do the job
        """
        return m[self.x - 128 - self.offset - self.rx: self.x + 128 - self.offset - self.rx,
               self.y - 128 - self.ry: self.y + 128 - self.ry]




class Crop256(object):
    def __init__(self,x,y,rx,ry):
        self.x = x
        self.y = y
        self.rx = rx
        self.ry = ry
        self.offset = 32

    def __call__(self, m):
        # if m.shape[0]<128:
        #     dummy = [m[-1] for i in range(128-m.shape[0])]
        #     dummy = np.array(dummy)
        #     m = np.concatenate((m, dummy), axis=0)

        """
        use the method of add 2d-image to make certain chunk of image stack
        """
        # return m[:128,
        #        self.x - 32 - self.offset - self.rx: self.x + 32 - self.offset - self.rx,
        #        self.y - 32 - self.ry: self.y + 32 - self.ry]

        """
        just return the color channel (slices), rely on the Resize to do the job
        """
        return m[:,
               self.x - 128 - self.offset - self.rx: self.x + 128 - self.offset - self.rx,
               self.y - 128 - self.ry: self.y + 128 - self.ry]




class ForegroundLabel(object):
    def __init__(self):
        pass
    def __call__(self, m):
        tm = np.zeros(m.shape)
        tm[m>=1] = 1

        return tm

class IntoTensor(object):
    def __init__(self, type='image'):
        self.type = type

    def __call__(self, m):
        #m = np.copy(m)
        if self.type == 'image':
            m = (m-m.mean())/m.std()
            return torch.tensor(m, dtype=torch.float32).unsqueeze(0)

        #print(np.unique(m))
        return torch.tensor(m, dtype=torch.float32).unsqueeze(0)


class CLAHE(object):
    '''
    apply for enhancing the image -1~1
    return enhanced image -1~1
    '''
    def __init__(self, kernel=None):
        self.kernel = kernel
    def __call__(self, m):
        m = (m-m.min())/(m.max()-m.min())
        # *t.max() 是为了让2d的图的最大值不会飘离实际
        return np.array([t.max() * equalize_adapthist(t,kernel_size=self.kernel) for t in m])


class LabelReduction(object):
    def __init__(self):
        pass
    def __call__(self, m):
        # (brain stem) (eye_L, eye_R) (Lens_L, Lens_R) (opt_nerve_L, opt_nerve)
        # opti_chiasma  (temporal_lobe_L,Temporal_Lobes_R
        # 1    2,3  45   67  8  9,10   11   12,13  14,15  16,17   18,19  20  21,22
        lb_l = [2, 4, 6, 9, 12, 14, 16, 18, 21]
        lb_r = [3, 5, 7, 10, 13, 15, 17, 19, 22]
        for idx, label_right in enumerate(lb_r):
            m[m==label_right] = lb_l[idx]
        # now label is
        # 1  2  4  6  8   9  11  12  14  16  18  20  21
        lb_old = [1,2,4,6,8,9,11,12,14,16,18,20,21]
        for idx, label_old in enumerate(lb_old):
            m[m==label_old] = idx+1

        return m


class NumpyNorm(object):
    def __init__(self):
        pass
    def __call__(self, m):
        m = (m-m.mean())/m.std()

        return m




if __name__ == '__main__':
    a = np.load('/mnt/HDD/datasets/HaN_OAR/train/12/label.npy')

    b = LabelReduction()(a)
    print(np.unique(b[90]))
    plot2d(b[90],bar=True)
    print(len(np.unique(b)))