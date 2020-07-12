import argparse
import os

from set.ds3d  import HaN_OAR_v2 as ProbSet
import random
from models.vnet import VNet
from tqdm import tqdm
from metrics import *
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch




@torch.no_grad()
def predict(config, test_on, is_train, fold):
    if config.model_type not in ['VNet',]:
        print('ERROR!! model_type should be selected in VNet/')
        print('Your input for model_type was %s' % config.model_type)
        return

    # #train_set = ProbSet(config.train_path)
    # valid_set = ProbSet(config.valid_path,is_train=False)
    test_set = ProbSet(config.test_path,is_train=is_train, is_aug=False, return_params=True, test_on=test_on, fold=fold)
    # print(len(valid_set), len(test_set))
    #train_loader = DataLoader(train_set, batch_size=config.batch_size)
    # valid_loader = DataLoader(valid_set, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, batch_size=config.batch_size)


    net = VNet()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    # print(config.model_type, net)

    net.load_state_dict(torch.load(config.net_path))
    net.eval()


    DC = 0.  # Dice Coefficient
    length = 0
    iou = 0
    for i, (imgs, gts, _, case) in enumerate(test_loader):

        #path = path[0] # 因为经过了loader被wrap进了元组 又因为batchsize=1
        case = case[0]

        imgs = imgs.to(device)
        gts = gts.round().long().to(device)

        outputs = net(imgs)
        print(gts.cpu().shape, imgs.shape, outputs.shape)
        # torch.Size([1, 1, 128, 128, 128]) torch.Size([1, 1, 128, 128, 128]) torch.Size([1, 14, 128, 128, 128])
        #print(path)
        ious = IoU(gts.detach().cpu().squeeze().numpy().reshape(-1),
                   outputs.detach().cpu().squeeze().argmax(dim=0).numpy().reshape(-1), num_classes=14)
        print(ious)
        print(np.array(ious).mean())
        iou += np.array(ious).mean()
        #print(path)
        #output_id = path.split('/')[-1]
        np.save('/mnt/EXTRA/datasets/competitions/aug/{}/{}/vnet-fold{}-z128-halved-clahe.npy'.format(TEST_ON,case,fold), outputs.detach().cpu().squeeze().numpy())
        print(case,outputs.detach().cpu().squeeze().numpy().shape)


        # for j in range(70,128):
        #     plt.figure()
        #     plt.subplot(2,2,1)
        #     # plt.imshow(np.array(imgs.cpu().squeeze()[j,0]))
        #     plt.imshow(np.array(imgs.cpu().squeeze()[j]))
        #     plt.colorbar()
        #     plt.subplot(2, 2, 2)
        #     plt.title(np.unique(np.array(gts.cpu().detach().numpy().squeeze()[j])))
        #     plt.imshow(np.array(gts.cpu().detach().numpy().squeeze()[j]))
        #     plt.colorbar()
        #     plt.subplot(2, 2, 3)
        #     plt.title(np.unique(outputs.cpu().detach().numpy().squeeze().argmax(axis=0)[j]))
        #     plt.imshow(outputs.cpu().detach().numpy().squeeze().argmax(axis=0)[j].reshape(128,128))
        #     #plt.imshow(outputs.cpu().detach().numpy().squeeze()[8,j].reshape(128, 128))
        #     plt.colorbar()
        #     plt.show()
        #     time.sleep(2)

    print('######', iou/10)
            # np.save('/mnt/HDD/datasets/HaN_OAR/image70.npy', np.array(imgs.cpu().squeeze()[i]))
            # np.save('/mnt/HDD/datasets/HaN_OAR/prediction70.npy', np.array(torch.sigmoid(outputs.cpu().detach()).numpy().squeeze()[i]))
    #     if config.output_ch == 1:
    #         outputs = torch.sigmoid(outputs)
    #
    #     acc += get_accuracy(outputs, gts) * imgs.size(0)
    #     SE += get_sensitivity(outputs, gts) * imgs.size(0)
    #     SP += get_specificity(outputs, gts) * imgs.size(0)
    #     PC += get_precision(outputs, gts) * imgs.size(0)
    #     F1 += get_F1(outputs, gts) * imgs.size(0)
    #     JS += get_JS(outputs, gts) * imgs.size(0)
    #     DC += get_DC(outputs, gts) * imgs.size(0)
    #     length += imgs.size(0)
    #
    # acc = acc / length
    # SE = SE / length
    # SP = SP / length
    # PC = PC / length
    # F1 = F1 / length
    # JS = JS / length
    # DC = DC / length
    # score = JS + DC
    # print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'
    #       % (acc, SE, SP, PC, F1, JS, DC))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters

    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    data_root = '/mnt/HDD/datasets/competitions/aug/'
    save_root = '/mnt/HDD/datasets/competitions/vnet/'
    parser.add_argument('--model_type', type=str, default='VNet', help='VNet/')
    parser.add_argument('--model_path', type=str, default=save_root + 'models_for_cls')
    parser.add_argument('--train_path', type=str, default=data_root)
    parser.add_argument('--valid_path', type=str, default=data_root)
    parser.add_argument('--test_path', type=str, default=data_root)
    parser.add_argument('--result_path', type=str, default=save_root + 'result_for_cls/')
    #  VNet-400-0.0001000-200-0.5000-vv-fold1-ce+dice.pkl
    #  VNet-100-0.0001000-50-0.5000-vv-fold1-1-ce+dice-then-gdice+ce-1.pkl
    # /mnt/HDD/datasets/competitions/candidate/vnet/VNet-400-0.0001000-100-0.5000-vv-fold2-1-ce+dice.pkl
    parser.add_argument('--net_path', type=str, default='/mnt/HDD/datasets/competitions/vnet/models_for_cls/VNet-60-0.0001000-25-0.5000-vv-fold5-d19-ce+dice-then-gdice+ce.pkl')

    parser.add_argument('--cuda_idx', type =int, default=1)

    config = parser.parse_args()

    # TEST_ON = 'right'
    # IS_TRAIN = True
    FOLD = 5

    for TEST_ON in ['raw', 'left', 'right']:
        for IS_TRAIN in [False, True]:
            predict(config,TEST_ON, IS_TRAIN, FOLD)