import argparse
import os

from set.ds2d import UnetSet
import random
from models.xnet import XNet
from tqdm import tqdm
from metrics import *
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch


@torch.no_grad()
def predict(config):
    if config.model_type not in ['UNet',]:
        print('ERROR!! model_type should be selected in UNet/')
        print('Your input for model_type was %s' % config.model_type)
        return


    valid_set = UnetSet(config.valid_path,is_train=False)
    test_set = UnetSet(config.test_path,is_train=False)
    # print(len(valid_set), len(test_set))
    #train_loader = DataLoader(train_set, batch_size=config.batch_size)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, batch_size=config.batch_size)


    net = XNet(num_classes=23)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    print(config.model_type, net)

    net.load_state_dict(torch.load(config.net_path))
    net.eval()


    DC = 0.  # Dice Coefficient
    length = 0
    iou = 0
    sk_o = []
    sk_g = []
    k=10
    for i, (imgs, probs, gts) in enumerate(test_loader):



        imgs = imgs.to(device)
        probs = probs.to(device)
        gts = gts.round().long().to(device)

        outputs = net(imgs, probs)
        #print(gts.cpu().shape, imgs.shape, outputs.shape)

        # ious = IoU(gts.detach().cpu().squeeze().numpy().reshape(-1),
        #            outputs.detach().cpu().squeeze().argmax(dim=0).numpy().reshape(-1), num_classes=14)
        # print(ious)
        # print(np.array(ious).mean())
        # iou += np.array(ious).mean()
        # print(path)
        # output_id = path.split('/')[-1]
        # np.save('/mnt/HDD/datasets/competitions/vnet/output/fold1/output{}.npy'.format(output_id), outputs.detach().cpu().squeeze().numpy())

        # if i >= (k-1)*128 and i <k*128:
        #     sk_o.append(outputs.detach().cpu().squeeze().argmax(dim=0).numpy())
        #     sk_g.append(gts.detach().cpu().squeeze().numpy())
        #     print(i)
        # #if (i+1)%(k*128) ==0:
        # if i==1279:
        #     sk_o = np.array(sk_o)
        #     sk_g = np.array(sk_g)
        #     ious = IoU(sk_g.reshape(-1), sk_o.reshape(-1), num_classes=23)
        #     print(ious)
        #     print(np.array(ious).mean())
        #     break

        plt.figure()
        plt.subplot(2,2,1)
        # plt.imshow(np.array(imgs.cpu().squeeze()[j,0]))
        plt.imshow(np.array(imgs.cpu().squeeze()))
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.title(np.unique(np.array(gts.cpu().detach().numpy().squeeze())))
        plt.imshow(np.array(gts.cpu().detach().numpy().squeeze()))
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.title(np.unique(outputs.cpu().detach().numpy().squeeze().argmax(axis=0)))
        plt.imshow(outputs.cpu().detach().numpy().squeeze().argmax(axis=0).reshape(128,128))
        #plt.imshow(outputs.cpu().detach().numpy().squeeze()[8,j].reshape(128, 128))
        plt.colorbar()
        plt.show()
        time.sleep(2)






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
    data_root = '/mnt/HDD/datasets/competitions/HaN_OAR/'
    save_root = '/mnt/HDD/datasets/competitions/vnet/'
    parser.add_argument('--model_type', type=str, default='UNet', help='UNet/')
    parser.add_argument('--model_path', type=str, default=save_root + 'models_for_cls')
    parser.add_argument('--train_path', type=str, default=data_root)
    parser.add_argument('--valid_path', type=str, default=data_root)
    parser.add_argument('--test_path', type=str, default=data_root)
    parser.add_argument('--result_path', type=str, default=save_root + 'result_for_cls/')
    parser.add_argument('--net_path', type=str, default='/mnt/HDD/datasets/competitions/unet/models_for_cls/UNet-100-0.0001000-20-0.5000-ce-100-20-unet-ce-fold1-23.pkl')

    parser.add_argument('--cuda_idx', type =int, default=1)

    config = parser.parse_args()
    predict(config)