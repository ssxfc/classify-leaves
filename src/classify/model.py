import torch.nn as nn
from torch.nn import functional as F


class ResNetblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetblock, self).__init__()
        self.blockconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels * 4)
        )
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        residual = x
        out = self.blockconv(x)
        if hasattr(self, 'shortcut'):  # 如果self中含有shortcut属性
            residual = self.shortcut(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=176):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2)
        )
        self.in_channels = 64
        # ResNet50中的四大层，每大层都是由ConvBlock与IdentityBlock堆叠而成
        self.layer1 = self.make_layer(ResNetblock, 64, 3, stride=1)
        self.layer2 = self.make_layer(ResNetblock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResNetblock, 256, 6, stride=2)
        self.layer4 = self.make_layer(ResNetblock, 512, 3, stride=2)

        self.avgpool = nn.AvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 4, num_classes)

    # 每个大层的定义函数
    def make_layer(self, block, channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * 4

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
