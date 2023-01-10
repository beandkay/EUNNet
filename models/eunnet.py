import torch
import torch.nn as nn
from .layers import Conv2d

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, block_idx, max_block,
                 stride=1, groups=1, base_width=64, drop_conv=0.0):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = Conv2d(inplanes, width, kernel_size=1, bias=True)
        self.conv2 = Conv2d(width, width, kernel_size=3, padding=1, stride=stride, groups=groups, bias=True)
        self.conv3 = Conv2d(width, planes * self.expansion, kernel_size=1, bias=True)
        self._scale = nn.Parameter(torch.ones(1))
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        self.residual = max_block ** -.5
        self.identity = block_idx ** .5 / (block_idx + 1) ** .5

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            avgpool = nn.AvgPool2d(stride) if stride != 1 else nn.Sequential()
            self.downsample = nn.Sequential(
                avgpool,
                Conv2d(inplanes, self.expansion*planes, kernel_size=1, bias=True))
            nn.init.kaiming_normal_(self.downsample[1].weight, nonlinearity='relu')
            nn.init.constant_(self.downsample[1].bias, 0)

        self.drop = nn.Sequential()
        if drop_conv > 0.0:
            self.drop = nn.Dropout2d(drop_conv)

    def forward(self, x):
        out = self.drop(self.conv1(x)).relu_()
        # out = out.mul(self._scale1)
        out = self.drop(self.conv2(out)).relu_()
        # out = out.mul(self._scale2)
        out = self.drop(self.conv3(out))

        out = out.mul(self._scale.mul(self.residual))
        # out = out.mul(self.residual)
        out = torch.add(input=out, alpha=self.identity, other=self.downsample(x))
        out = out.relu_()
        print(torch.nonzero(out).size(0) / out.numel())
        return out


class EUNNet(nn.Module):

    def __init__(self, layers, num_classes=1000, groups=1, width_per_group=64,
                drop_conv=0.0, drop_fc=0.0):
        super(EUNNet, self).__init__()
        block = Bottleneck

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.block_idx = sum(layers) - 1
        self.max_depth = sum(layers)

        self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_conv=drop_conv)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, drop_conv=drop_conv)
        self.drop = nn.Dropout(drop_fc)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        i = 1

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc.weight, a=1)

    def _make_layer(self, block, planes, num_blocks, stride=1, drop_conv=0.0):
        strides = [stride] + [1] * (num_blocks -1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, block_idx=self.block_idx, max_block=self.max_depth,
                stride=stride, groups=self.groups, base_width=self.base_width, drop_conv=drop_conv))
            self.inplanes = planes * block.expansion
            self.block_idx += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean((2,3))
        x = self.drop(x)
        x = self.fc(x)
        return x

def eunnet50(num_classes=1000, drop_conv=0.0, drop_fc=0.0):
    return EUNNet([3, 4, 6, 3], num_classes=num_classes,
        drop_conv=drop_conv, drop_fc=drop_fc, groups=1, width_per_group=64)

def eunnet101(num_classes=1000, drop_conv=0.0, drop_fc=0.0):
    return EUNNet([3, 4, 23, 3], num_classes=num_classes,
        drop_conv=drop_conv, drop_fc=drop_fc, groups=1, width_per_group=64)

def eunnet200(num_classes=1000, drop_conv=0.0, drop_fc=0.0):
    return EUNNet([3,24,36,3], num_classes=num_classes,
        drop_conv=drop_conv, drop_fc=drop_fc, groups=1, width_per_group=64)

def eunnetX50_32x4d(num_classes=1000, drop_conv=0.0, drop_fc=0.0):
    return EUNNet([3, 4, 6, 3], num_classes=num_classes,
        drop_conv=drop_conv, drop_fc=drop_fc, groups=32, width_per_group=4)

def eunnetX101_32x8d(num_classes=1000, drop_conv=0.0, drop_fc=0.0):
    return EUNNet([3, 4, 23, 3], num_classes=num_classes,
        drop_conv=drop_conv, drop_fc=drop_fc, groups=32, width_per_group=8)
