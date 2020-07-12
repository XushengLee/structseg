import torch
import torch.nn as nn
import torch.nn.functional as F
# SR : Segmentation Result
# GT : Ground Truth
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial import KDTree


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1) + (GT == 1)) == 2
    FN = ((SR == 0) + (GT == 1)) == 2

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0) + (GT == 0)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1) + (GT == 1)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR + GT) == 2)
    Union = torch.sum((SR + GT) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR + GT) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC


def to_one_hot(tensor, nClasses):
    b, h, w = tensor.size()
    one_hot = torch.zeros(b, nClasses,h,w).scatter_(1, tensor.view(b,1,h,w),1)

    return one_hot


# class mIoULoss(nn.Module):
#     def __init__(self, weight=None, size_average=True, n_classes=2):
#         super(mIoULoss, self).__init__()
#         self.classes = n_classes
#
#     def forward(self, inputs, target_oneHot):
#         # inputs => N x Classes x H x W
#         # target_oneHot => N x Classes x H x W
#
#         N = inputs.size()[0]
#
#         # predicted probabilities for each pixel along channel
#         inputs = F.softmax(inputs, dim=1)
#
#         # Numerator Product
#         inter = inputs * target_oneHot
#         ## Sum over all pixels N x C x H x W => N x C
#         inter = inter.view(N, self.classes, -1).sum(2)
#
#         # Denominator
#         union = inputs + target_oneHot - (inputs * target_oneHot)
#         ## Sum over all pixels N x C x H x W => N x C
#         union = union.view(N, self.classes, -1).sum(2)
#
#         loss = inter / union
#
#         ## Return average loss over classes and batch
#         return 1 - loss.mean()



def IoU(gt, pred, num_classes=23):
    SMOOTH = 1e-6
    cm = confusion_matrix(gt, pred)
    ious = []
    for idx in range(1, min(num_classes, len(cm))):
        # ious.append(float(cm[idx,idx]) / float(cm[idx,:].sum() + cm[:,idx].sum() - cm[idx, idx]))
        ious.append(float(2*cm[idx, idx]) / float(cm[idx, :].sum() + cm[:, idx].sum()))

    return ious


def dis_from_a2b(a,b):
    kdTree = KDTree(a, leafsize=100)
    return kdTree.query(b, k=1,eps=0,p=2)[0]

def HD_dis(gt, pred, num_classes=23):
    h_dist = []
    for k in range(1, num_classes):
        gt_val = np.reshape(np.where)




if __name__ == '__main__':
    a = np.array([
        [2,0,0,0,0],
        [0,1,1,1,1],
        [0,1,1,1,1],
        [0,1,1,1,1],
        [0,0,0,0,0]
    ])
    b = np.array([
        [2,2,2,0,0],
        [1,1,1,1,0],
        [1,1,1,1,0],
        [1,1,1,1,0],
        [0,0,0,0,0]
    ])
    cm = confusion_matrix(a.reshape(-1),b.reshape(-1))
    print(cm)
    print(float(cm[2,2])/float(cm[2,:].sum()+cm[:,2].sum()-cm[2,2]))
    print(IoU(a.reshape(-1), b.reshape(-1), 3))








