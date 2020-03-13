'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys
sys.path.append("/home/weihao_zhuang/workspace/group_equivariant_conv/")

from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M

def P4MConvZ2_3x3(in_planes, planes, kernel_size=3,stride=1,padding=1,bias=False):
    return P4MConvZ2(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def P4MConvZ2_5x5(in_planes, planes, kernel_size=5,stride=1,padding=2,bias=False):
    return P4MConvZ2(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def P4MConvP4M_3x3(in_planes, planes, kernel_size=3,stride=1,padding=1,bias=False):
    return P4MConvP4M(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def P4MConvP4M_5x5(in_planes, planes, kernel_size=5,stride=1,padding=2,bias=False):
    return P4MConvP4M(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def P4MConvP4M_1x1(in_planes, planes, kernel_size=1,stride=1,padding=0,bias=False):
    return P4MConvP4M(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

conv_p4m = P4MConvP4M_3x3
conv_z2 = P4MConvZ2_3x3

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_p4m(in_planes, planes, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv_p4m(planes, planes, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4MConvP4M_1x1(in_planes, self.expansion*planes, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = P4MConvP4M_1x1(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv_p4m(planes, planes, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = P4MConvP4M_1x1(planes, self.expansion*planes, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4MConvP4M_1x1(in_planes, self.expansion*planes, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, conv_type = '3x3'):
        super(ResNet, self).__init__()
        
        global conv_z2
        global conv_p4m
        if conv_type == '3x3':
            conv_p4m = P4MConvP4M_3x3
            conv_z2 = P4MConvZ2_3x3
        elif conv_type == '5x5':
            conv_p4m = P4MConvP4M_5x5
            conv_z2 = P4MConvZ2_5x5
        
        self.in_planes = 23

        self.conv1 = conv_z2(3, 23, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(23)
        self.layer1 = self._make_layer(block, 23, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 45, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 91, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 181, num_blocks[3], stride=2)
        self.linear = nn.Linear(181*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        outs = out.size()
        out = out.view(outs[0], outs[1]*outs[2], outs[3], outs[4])
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(conv_type='3x3'):
    return ResNet(BasicBlock, [2,2,2,2],conv_type=conv_type)

def ResNet34(conv_type='3x3'):
    return ResNet(BasicBlock, [3,4,6,3],conv_type=conv_type)

def ResNet50(conv_type='3x3'):
    return ResNet(Bottleneck, [3,4,6,3],conv_type=conv_type)

def ResNet101(conv_type='3x3'):
    return ResNet(Bottleneck, [3,4,23,3],conv_type=conv_type)

def ResNet152(conv_type='3x3'):
    return ResNet(Bottleneck, [3,8,36,3],conv_type=conv_type)


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
