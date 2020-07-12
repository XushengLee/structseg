import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_feat, out_feat,
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_feat),
            nn.PReLU(out_feat),
        )

    def forward(self, x):
        return self.conv(x)



class EncoderBlock(nn.Module):
    def __init__(self, in_f, out_f):
        super(EncoderBlock, self).__init__()
        self.conv_block1 = ConvBlock(in_feat=in_f, out_feat=in_f)
        self.conv_block2 = ConvBlock(in_feat=in_f, out_feat=in_f)
        self.downsample = nn.Conv2d(in_channels=out_f, out_channels=out_f,
                                    kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x1)
        # torch.cat([x,x2], dim=1)
        return self.downsample(torch.cat([x,x2], dim=1))



class DencoderBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(DencoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_feat, out_feat,
                                         kernel_size=2,
                                         stride=2)
        self.conv_block1 = ConvBlock(in_feat, out_feat)
        self.conv_block2 = ConvBlock(out_feat, out_feat)

    def forward(self, x, down_x):
        x = self.deconv(x)

        x = torch.cat([down_x, x], dim=1)
        return self.conv_block2(self.conv_block1(x))


class BottomBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(BottomBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_feat, out_feat,
                      kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_feat),
            nn.PReLU(out_feat),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_feat, out_feat,
                      kernel_size=3, stride=1,
                      dilation=2, padding=2),
            nn.BatchNorm2d(out_feat),
            nn.PReLU(out_feat),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(out_feat, out_feat,
                      kernel_size=3, stride=1,
                      dilation=5, padding=5),
            nn.BatchNorm2d(out_feat),
            nn.PReLU(out_feat),
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        return x1 + x2 + x3


class DRUNet(nn.Module):
    def __init__(self, num_channels=1, num_classes = 23):
        super(DRUNet, self).__init__()
        in_conv_feat = 32
        num_feat = [64, 128, 256, 512, 1024]

        self.in_conv = ConvBlock(num_channels, in_conv_feat) # 1   64
        self.down1 = EncoderBlock(in_conv_feat, num_feat[0]) # 64  64
        self.down2 = EncoderBlock(num_feat[0], num_feat[1]) # 64  128
        self.down3 = EncoderBlock(num_feat[1], num_feat[2]) # 128 256
        self.down4 = EncoderBlock(num_feat[2], num_feat[3]) # 256 512
        self.bottom = BottomBlock(num_feat[3], num_feat[4]) # 512 1024
        self.up1 = DencoderBlock(num_feat[4], num_feat[3])
        self.up2 = DencoderBlock(num_feat[3], num_feat[2])
        self.up3 = DencoderBlock(num_feat[2], num_feat[1])
        self.up4 = DencoderBlock(num_feat[0], num_feat[0])

        self.final = nn.Sequential(
            nn.Conv2d(num_feat[0],num_classes,
                      kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):

        x = self.in_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        b = self.bottom(x4)

        up1 = self.up1(b, x4)
        up2 = self.up2(up1, x3)
        up3 = self.up3(up2, x2)
        up4 = self.up4(up3, x1)

        return self.final(up4)



class DRUNetSmall(nn.Module):
    def __init__(self, num_channels=14, num_classes = 23):
        super(DRUNetSmall, self).__init__()
        num_feat = [64, 128, 256, 512]

        self.in_conv = ConvBlock(num_channels, num_feat[0]) # 1   64
        self.down1 = EncoderBlock(num_feat[0], num_feat[0]) # 64  64
        self.down2 = EncoderBlock(num_feat[0], num_feat[1]) # 64  128
        self.down3 = EncoderBlock(num_feat[1], num_feat[2]) # 128 256

        self.bottom = BottomBlock(num_feat[2], num_feat[3]) # 512 1024

        self.up1 = DencoderBlock(num_feat[3], num_feat[2])
        self.up2 = DencoderBlock(num_feat[2], num_feat[1])
        self.up3 = DencoderBlock(num_feat[1], num_feat[0])


        self.final = nn.Sequential(
            nn.Conv2d(num_feat[0],num_classes,
                      kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):

        x = self.in_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        b = self.bottom(x3)

        up1 = self.up2(b, x3)
        up2 = self.up3(up1, x2)
        up3 = self.up4(up2, x1)

        return self.final(up3)



class XDRUNet(nn.Module):
    def __init__(self, unet_num_c =1, small_unet_num_c=14, n_classes=23):
        super(XDRUNet, self).__init__()
        self.drunet = DRUNet(unet_num_c, n_classes)
        self.drunet_small = DRUNetSmall(small_unet_num_c, n_classes)

        self.bn_relu1 = nn.Sequential(
            nn.BatchNorm2d(n_classes),
            nn.PReLU(n_classes)
        )

        self.bn_relu2 = nn.Sequential(
            nn.BatchNorm2d(n_classes),
            nn.PReLU(n_classes)
        )

        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n_classes,out_channels=n_classes,
                               kernel_size=2,stride=2),
            nn.Conv2d(in_channels=n_classes, out_channels=n_classes,
                      kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(n_classes),
            nn.Sigmoid(),
        )

    def forward(self, x, prob):
        x = self.drunet(x)
        prob = self.upscale(self.drunet_small(prob))

        return x*prob