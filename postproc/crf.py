import numpy as np
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

from plots import plot2d

#--------- 对于这种没有明显轮廓的图效果不升反降 而且有虚假的噪点 有效果的部分 估计是 probs图中的点概率也不高

def CRFs(image, prediction, num_iter=10, sdims=(10,10), schan=(0.01,), compat=10):
    """

    :param image:
    :param prediction:
    :return:
    """
    W, H = image.shape
    NLABELS = 2  # 1 class plus 1 background
    probs = np.stack([1-prediction, prediction], axis=0)
    U = unary_from_softmax(probs)  # note: num classes is first dim


    # Create the pairwise bilateral term from the above image.
    # The two `s{dims,chan}` parameters are model hyper-parameters defining
    # the strength of the location and image content bilaterals, respectively.
    pairwise_energy = create_pairwise_bilateral(sdims=sdims, schan=schan, img=image.reshape(W,H,1), chdim=2)

    d = dcrf.DenseCRF2D(W, H, NLABELS)
    d.setUnaryEnergy(U)
    d.addPairwiseEnergy(pairwise_energy, compat=compat)  # `compat` is the "strength" of this potential.

    # This time, let's do inference in steps ourselves
    # so that we can look at intermediate solutions
    # as well as monitor KL-divergence, which indicates
    # how well we have converged.
    # PyDenseCRF also requires us to keep track of two
    # temporary buffers it needs for computations.
    Q, tmp1, tmp2 = d.startInference()
    for _ in range(num_iter):
        d.stepInference(Q, tmp1, tmp2)

    # kl divergence, but is negative, dont know why
    # kl1 = d.klDivergence(Q) / (H * W)
    map_soln = np.argmax(Q, axis=0).reshape((H, W))

    # And let's have a look.
    return  map_soln


if __name__ == '__main__':
    image = np.load('/mnt/HDD/datasets/HaN_OAR/image70.npy')
    prediction = np.load('/mnt/HDD/datasets/HaN_OAR/prediction70.npy')
    map = CRFs(image, prediction.round())
    plot2d(prediction)
    plot2d(map, bar=True)
    '''
    CRF 在这个任务上表现并不好 但只是是输入未round的特征图的情况
    update: round后改变很小 几乎不变 感觉没有变化
    '''