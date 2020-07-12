# 改一个 mul-class的unet出来 先看看有没有用
# 尝试对 image， output 加入噪声 和 旋转 剪切什么的
# 先尝试用CE+DICE
# 尝试loss func 最好改一个mul-class的focal loss出来
import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from metrics import *
from models.vnet import VNet #, R2U_Net, AttU_Net
#from models.u_net import U_Net

from losses.dice import DiceLoss, expand_as_one_hot
from losses.focal import FocalLoss
import csv
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from set.ds3d import HaN_OAR_v2 as Probset3d
from torch.backends import cudnn
import random

import warnings
warnings.filterwarnings("ignore")

class Solver(object):
    def __init__(self, args, train_loader, val_loader, test_loader):
        # data loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # models
        self.net = None
        self.optimizer = None

        self.criterion = FocalLoss(alpha=0.8,gamma=0.5)  # torch.nn.BCELoss()
        self.augmentation_prob = args.augmentation_prob

        # hyper-param
        self.lr = args.lr
        self.decayed_lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        # training settings
        self.num_epochs = args.num_epochs
        self.num_epochs_decay = args.num_epochs_decay
        self.batch_size = args.batch_size

        # step size for logging and val
        self.log_step = args.log_step
        self.val_step = args.val_step

        # path
        self.best_score = 0
        self.best_epoch = 0
        self.model_path = args.model_path
        self.csv_path = args.result_path
        self.model_type = args.model_type

        self.comment = args.comment

        self.net_path = os.path.join(
            self.model_path, '%s-%d-%.7f-%d-%.4f-%s.pkl' %
                             (self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob,self.comment)
        )

        ########### TO DO multi GPU setting ##########
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



        self.build_model()




    def build_model(self):
        if self.model_type == 'VNet':
            ###### to do ########
            self.net = VNet()


        self.optimizer = optim.Adam(self.net.parameters(), self.lr, [self.beta1, self.beta2])
        self.net.to(self.device)

        #self.print_network(self.net, self.model_type)

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  # numel() return total num of elems in tensor
        print(model)
        print(name)
        print('the number of parameters: {}'.format(num_params))

    # =============================== train =========================#
    # ===============================================================#
    def train(self, epoch):
        self.net.train(True)

        # Decay learning rate
        if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
            self.decayed_lr -= (self.lr / float(self.num_epochs_decay))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.decayed_lr
            print('epoch{}: Decay learning rate to lr: {}.'.format(epoch, self.decayed_lr))

        epoch_loss = 0

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        length = 0

        for i, (imgs, gts) in enumerate(tqdm(self.train_loader)):
            imgs = imgs.to(self.device)
            gts = gts.round().long().to(self.device)

            self.optimizer.zero_grad()

            outputs = self.net(imgs)

            # make sure shapes are the same by flattening them

            # weight = torch.tensor([1.,100.,100.,100.,50.,50.,80., 80.,50.,80.,80.,80.,50.,50.,70.,70.,70.,70.,
            #                        60.,60.,100.,100.,100.,]).to(self.device)


            # weights as demanded
            # weight = torch.tensor(
            #     [1., 100., 100., 50., 80., 50., 80., 80., 50., 70., 70.,
            #      60., 100., 100., ]).to(self.device)

            # self cal

            weight = torch.tensor(
                [1. ,100., 130., 1000., 700., 900., 30., 1000., 60., 200.,100.,300.,100.,55.]).to(self.device)

            ce_loss = nn.CrossEntropyLoss(weight=weight,reduction='mean')(outputs, gts.reshape(-1,128,128,128))
            dice_loss = DiceLoss(sigmoid_normalization=False)(outputs, expand_as_one_hot(gts.reshape(-1,128,128,128),14))
            # bce_loss = torch.nn.BCEWithLogitsLoss()(outputs, gts)
            # focal_loss = FocalLoss(alpha=0.8,gamma=0.5)(outputs, gts)

            loss = dice_loss + ce_loss
            #loss = focal_loss + dice_loss
            epoch_loss += loss.item() * imgs.size(0)  # because reduction = 'mean'
            loss.backward()
            self.optimizer.step()


            # DC += iou(outputs.detach().cpu().squeeze().argmax(dim=1),gts.detach().cpu(),n_classes=14)*imgs.size(0)
            # length += imgs.size(0)



        # DC = DC / length
        # epoch_loss = epoch_loss/length
        # # Print the log info
        # print(
        #     'Epoch [%d/%d], Loss: %.4f, \n[Training] DC: %.4f' % (
        #         epoch + 1, self.num_epochs,
        #         epoch_loss,
        #          DC))
        print('EPOCH{}'.format(epoch))

    # =============================== validation ====================#
    # ===============================================================#
    @torch.no_grad()
    def validation(self, epoch):
        self.net.eval()

        acc = 0.  # Accuracy
        SE = 0.  # Sensit ivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        length = 0

        for i, (imgs, gts) in enumerate(self.val_loader):
            imgs = imgs.to(self.device)
            gts = gts.round().long().to(self.device)

            outputs = self.net(imgs)



            weight = np.array(
                [0., 100., 100., 100., 50., 50., 80., 80., 50., 80., 80., 80., 50., 50., 70., 70., 70., 70.,
                 60., 60., 100., 100., 100., ])
            ious = IoU(gts.detach().cpu().squeeze().numpy().reshape(-1), outputs.detach().cpu().squeeze().argmax(dim=0).numpy().reshape(-1),num_classes=14)*imgs.size(0)
            DC += np.array(ious[1:]).mean()
            length += imgs.size(0)


        DC = DC / length

        score = DC

        print('[Validation] DC: %.4f' % (
            DC))

        # save the best net model
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            print('Best %s model score: %.4f'%(self.model_type, self.best_score))
            torch.save(self.net.state_dict(), self.net_path)
        # if (1+epoch)%10 == 0 or epoch==0:
        #     torch.save(self.net.state_dict(), self.net_path+'epoch{}.pkl'.format(epoch))
        # if (epoch+1)%50 == 0 and epoch!=1:
        #     torch.save(self.net.state_dict(),
        #                '/mnt/HDD/datasets/competitions/vnet/models_for_cls/400-200-dice-epoch{}.pkl'.format(epoch+1))

    def test(self):
        del self.net
        self.build_model()
        self.net.load_state_dict(torch.load(self.net_path))

        self.net.eval()

        DC = 0.  # Dice Coefficient
        length = 0

        for i, (imgs, gts) in enumerate(self.test_loader):
            imgs = imgs.to(self.device)
            gts = gts.round().long().to(self.device)

            outputs = self.net(imgs)

            weight = np.array(
                [0., 100., 100., 100., 50., 50., 80., 80., 50., 80., 80., 80., 50., 50., 70., 70., 70., 70.,
                 60., 60., 100., 100., 100., ])
            ious = IoU(gts.detach().cpu().squeeze().numpy().reshape(-1),
                       outputs.detach().cpu().squeeze().argmax(dim=0).numpy().reshape(-1), num_classes=14) * imgs.size(
                0)
            DC += np.array(ious[1:]).mean()
            length += imgs.size(0)


        DC = DC / length
        score = DC


        f = open(os.path.join(self.csv_path, 'result.csv'), 'a', encoding='utf8', newline='')
        wr = csv.writer(f)
        wr.writerow([self.model_type, DC,
                     self.lr, self.best_epoch, self.num_epochs,
                     self.num_epochs_decay, self.augmentation_prob, self.batch_size, self.comment])
        f.close()


    def train_val_test(self):

        ################# BUG
        # if os.path.isfile(self.net_path):
        #     #self.net.load_state_dict(torch.load(self.net_path))
        #     print('saved {} is loaded form: {}'.format(self.model_type, self.net_path))
        # else:
        for epoch in range(self.num_epochs):
            self.train(epoch)
            self.validation(epoch)

        self.test()


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['VNet',]:
        print('ERROR!! model_type should be selected in VNet/')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    if config.random_hyperparam_search:
        lr = random.random() * 0.0005 + 0.0000005
        augmentation_prob = random.random() * 0.7
        epoch = random.choice([100, 150, 200, 250])
        decay_ratio = random.random() * 0.8
        decay_epoch = int(epoch * decay_ratio)

        config.augmentation_prob = augmentation_prob
        config.num_epochs = epoch
        config.lr = lr
        config.num_epochs_decay = decay_epoch

    print(config)

    train_set = Probset3d(config.train_path,is_train=True, fold=3)
    valid_set = Probset3d(config.valid_path,is_train=False,fold=3 )
    test_set = Probset3d(config.test_path,is_train=False, fold=3)

    train_loader = DataLoader(train_set, batch_size=config.batch_size,num_workers=config.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size,num_workers=config.num_workers)
    test_loader = DataLoader(test_set, batch_size=config.batch_size,num_workers=config.num_workers)

    # train_loader = get_loader(image_path=config.train_path,
    #                           batch_size=config.batch_size,
    #                           num_workers=config.num_workers,
    #                           is_train=True,
    #                           augmentation_prob=0.)
    #
    # valid_loader = get_loader(image_path=config.valid_path,
    #                           batch_size=config.batch_size,
    #                           num_workers=config.num_workers,
    #                           is_train=False,
    #                           augmentation_prob=0.)
    #
    # test_loader = get_loader(image_path=config.test_path,
    #                          batch_size=config.batch_size,
    #                          num_workers=config.num_workers,
    #                          is_train=False,
    #                          augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)


    solver.train_val_test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters

    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=400)  # random hyperparam search
    parser.add_argument('--num_epochs_decay', type=int, default=200)  # random hyperparam search
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--random-search', dest='random_hyperparam_search', action='store_true')
    parser.add_argument('--no-random-search', dest='random_hyperparam_search', action='store_false')
    parser.set_defaults(random_hyperparam_search=False)
    parser.add_argument('--lr', type=float, default=0.0001)  # random hyperparam search
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.5)  # random hyperparam search

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    data_root = '/mnt/HDD/datasets/competitions/aug/'
    save_root = '/mnt/HDD/datasets/competitions/vnet/'
    parser.add_argument('--model_type', type=str, default='VNet', help='VNet/')
    parser.add_argument('--model_path', type=str, default=save_root+'models_for_cls')
    parser.add_argument('--train_path', type=str, default=data_root)
    parser.add_argument('--valid_path', type=str, default=data_root)
    parser.add_argument('--test_path', type=str, default=data_root)
    parser.add_argument('--result_path', type=str, default=save_root+'result_for_cls/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    parser.add_argument('--comment', type=str, default='vv-fold3-3-wce+dice')

    config = parser.parse_args()
    main(config)




