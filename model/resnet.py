
import torch.nn as nn
from .group_op import E4_C4, C_4_Conv, C_4_BN, C_4_1x1, C_4_Pool, C_4_3x3, \
    E4_D4, D_4_Conv, D_4_BN, D_4_1x1, D_4_Pool, D_4_3x3


class BasicBlock(nn.Module):
    """Basic Block for resnet18
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, dropout, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.Dropout(dropout),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))



class C4Basic(nn.Module):
    """Basic Block for C4resnet18
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel, reduction, groups, dropout, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            C_4_3x3(in_channels, out_channels, stride=stride),
            nn.Dropout(dropout),
            C_4_BN(out_channels),
            nn.ReLU(inplace=True),
            C_4_3x3(out_channels, out_channels * BasicBlock.expansion),
            C_4_BN(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                C_4_1x1(in_channels, out_channels * BasicBlock.expansion, stride=stride),
                C_4_BN(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class D4Basic(nn.Module):
    """Basic Block for D4resnet18
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, reduction, kernel, groups, bias, dropout, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            D_4_3x3(in_channels, out_channels, stride=stride),
            nn.Dropout(dropout),
            D_4_BN(out_channels),
            nn.ReLU(inplace=True),
            D_4_3x3(out_channels, out_channels * BasicBlock.expansion, bias=bias),
            D_4_BN(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                D_4_1x1(in_channels, out_channels * BasicBlock.expansion, stride=stride),
                D_4_BN(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class E4C4_Basic(nn.Module):
    """Basic Block for E4C4resnet18
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel, reduction, groups, dropout, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            C_4_3x3(in_channels, out_channels, stride=stride),
            nn.Dropout(dropout),
            C_4_BN(out_channels),
            nn.ReLU(inplace=True),
            E4_C4(out_channels, out_channels * BasicBlock.expansion, kernel, reduction, groups=groups),
            C_4_BN(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                C_4_1x1(in_channels, out_channels * BasicBlock.expansion, stride=stride),
                C_4_BN(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class E4D4_Basic(nn.Module):
    """Basic Block for E4D4resnet18
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, reduction, kernel, groups, bias, dropout, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            D_4_3x3(in_channels, out_channels, stride=stride),
            nn.Dropout(dropout),
            D_4_BN(out_channels),
            nn.ReLU(inplace=True),
            E4_D4(out_channels, out_channels * BasicBlock.expansion, kernel, reduction, groups=groups),
            D_4_BN(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                D_4_1x1(in_channels, out_channels * BasicBlock.expansion, stride=stride),
                D_4_BN(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))



class ResNet(nn.Module):

    def __init__(self, block, num_block, dropout=0., num_classes=100):
        super().__init__()

        self.in_channels = 64
        self.dropout = dropout

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, self.dropout, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class C4ResNet(nn.Module):

    def __init__(self, block, num_block, kernel=3, reduction=2, groups=2, dropout=0., num_classes=100):
        super().__init__()

        self.in_channels = 32
        self.dropout = dropout
        self.kernel = kernel
        self.groups = groups
        self.reduction = reduction

        self.conv1 = nn.Sequential(
            C_4_Conv(3, 32),
            C_4_BN(32),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 32, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 64, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 128, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 256, num_block[3], 2)
        self.group_pool = C_4_Pool()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_channels, out_channels, self.kernel, self.reduction, self.groups, self.dropout, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.group_pool(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class D4ResNet(nn.Module):

    def __init__(self, block, num_block, reduction=2, kernel=3, groups=2, bias=False, dropout=0., num_classes=100):
        super().__init__()

        self.in_channels = 22
        self.dropout = dropout
        self.kernel = kernel
        self.groups = groups
        self.reduction = reduction
        self.bias = bias

        self.conv1 = nn.Sequential(
            D_4_Conv(3, 22),
            D_4_BN(22),
            nn.ReLU(inplace=True))
        # we use different inputsize of the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 22, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 44, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 88, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 176, num_block[3], 2)
        self.group_pool = D_4_Pool()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(176 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_channels, out_channels, self.reduction, self.kernel, self.groups, self.bias, self.dropout,
                      stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.group_pool(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output




def resnet18(dropout, num_classes):
    """ return a ResNet18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], dropout, num_classes=num_classes)


def C4resnet18(dropout, num_classes):
    """ return a C4ResNet18 object
    """
    return C4ResNet(C4Basic, [2, 2, 2, 2], dropout=dropout, num_classes=num_classes)


def E4C4resnet18(dropout, kernel, reduction, groups, num_classes):
    """ return a E4C4ResNet18 object
    """
    return C4ResNet(E4C4_Basic, [2, 2, 2, 2], kernel=kernel, reduction=reduction, groups=groups, dropout=dropout,
                    num_classes=num_classes)

def D4resnet18(dropout, bias, num_classes):
    """ return a D4ResNet18 object
    """
    return D4ResNet(D4Basic, [2, 2, 2, 2], bias=bias, dropout=dropout, num_classes=num_classes)



def E4D4resnet18(dropout, reduction, kernel, groups, num_classes):
    """ return a D4E4ResNet 18 object
    """
    return D4ResNet(E4D4_Basic, [2, 2, 2, 2], reduction, kernel, groups, dropout=dropout, num_classes=num_classes)

#Right

