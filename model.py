from torch import nn
import torch
import pdb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.module2 = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        self.module3 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1),


        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_block21 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block22 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block23 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block24 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block25 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block26 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, ms,pan):

        ms = torch.nn.functional.interpolate(ms, size=(pan.shape[2], pan.shape[3]), mode='bilinear')
        x1 = torch.cat([pan, ms], dim=1)

        x1 = self.module1(x1)
        y1 = self.conv_block21(pan)
        y21 = self.conv_block22(y1)
        y2 = torch.sub(y21, y1)

        y31 = self.conv_block23(y21)
        y3 = torch.sub(y31, y21)
        y3 = torch.sub(y3, y1)

        y41 = self.conv_block24(y31)
        y4 = torch.sub(y41, y31)
        y4 = torch.sub(y4, y21)
        y4 = torch.sub(y4, y1)

        y51 = self.conv_block25(y41)
        y5 = torch.sub(y51, y41)
        y5 = torch.sub(y5, y31)
        y5 = torch.sub(y5, y21)
        y5 = torch.sub(y5, y1)

        y61 = self.conv_block26(y51)
        y6 = torch.sub(y61, y51)
        y6 = torch.sub(y6, y41)
        y6 = torch.sub(y6, y31)
        y6 = torch.sub(y6, y21)
        y6 = torch.sub(y6, y1)

        x2 = self.conv_block1(x1)
        x2 = torch.add(x2, y1)
        x3 = torch.cat([x1, x2], dim=1)

        x4 = self.conv_block2(x3)
        x4 = torch.add(x4, y2)
        x5 = torch.cat([x3, x4], dim=1)

        x6 = self.conv_block3(x5)
        x6 = torch.add(x6, y3)
        x7 = torch.cat([x5, x6], dim=1)

        x8 = self.conv_block4(x7)
        x8 = torch.add(x8, y4)
        x9 = torch.cat([x7, x8], dim=1)

        x10 = self.conv_block5(x9)
        x10 = torch.add(x10, y5)
        x11 = torch.cat([x9, x10], dim=1)

        x12 = self.conv_block6(x11)
        x12 = torch.add(x12, y6)
        x13 = torch.cat([x11, x12], dim=1)

        x13 = self.module2(x13)
        x14 = self.module3(x13)
        for i in range(2):
            pan = torch.cat([pan, pan], dim=1)
        x14 = torch.add(x14, pan)
        return x14

class senet(nn.Module):
    def __init__(self, opt):
        super(senet, self).__init__()

        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.module2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = x
        x = self.module1(x)
        se = x
        x = self.se(x)
        x = torch.mul(x, se)
        x = self.module2(x)
        x =x * x1
        return x


class Generator1(nn.Module):
    def __init__(self, opt):
        super(Generator1, self).__init__()

        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.module2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.module3 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )


        self.conv_block21 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block22 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block23 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_block24 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )


    def forward(self, ms,pan):

        ms = torch.nn.functional.interpolate(ms, size=(pan.shape[2], pan.shape[3]), mode='bilinear')
        x1 = torch.cat([pan, ms], dim=1)

        x1 = self.module1(x1)
        y1 = self.conv_block21(pan)
        y21 = self.conv_block22(y1)
        y2 = torch.sub(y21, y1)

        y31 = self.conv_block23(y21)
        y3 = torch.sub(y31, y21)
        y3 = torch.sub(y3, y1)

        y41 = self.conv_block24(y31)
        y4 = torch.sub(y41, y31)
        y4 = torch.sub(y4, y21)
        y4 = torch.sub(y4, y1)

        x2 = self.conv_block1(x1)
        x2 = torch.add(x2, y1)
        x3 = torch.cat([x1, x2], dim=1)

        x4 = self.conv_block2(x3)
        x4 = torch.add(x4, y2)
        x5 = torch.cat([x3, x4], dim=1)

        x6 = self.conv_block3(x5)
        x6 = torch.add(x6, y3)
        x7 = torch.cat([x5, x6], dim=1)

        x8 = self.conv_block4(x7)
        x8 = torch.add(x8, y4)
        x9 = torch.cat([x7, x8], dim=1)

        x13 = self.module2(x9)
        x14 = self.module3(x13)

        for i in range(2):
            pan = torch.cat([pan, pan], dim=1)

        x19 = torch.add(x14, pan)

        return x19
