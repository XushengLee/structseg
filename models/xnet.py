import torch.nn as nn
import torch
from models.unet_zoo import *


class XNet(nn.Module):
    def __init__(self, img_c=1, mask_c=14, num_classes=14):
        super(XNet, self).__init__()
        self.unet = UNet(num_channels=img_c, num_classes=num_classes)
        self.smallunet = UNetSmall(num_channels=mask_c, num_classes=num_classes)

        self.final = nn.Sequential(nn.Conv2d(64,
                                             num_classes,
                                             kernel_size=1))

    def forward(self, x, mask, return_feature=False):
        x = self.unet(x, return_features=True)
        mask = self.smallunet(mask, return_features=True)


        return self.final(x*mask)