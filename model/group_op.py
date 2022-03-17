import torch.nn as nn
import torch
import math


class C_4_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(C_4_1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight = torch.randn(out_channels, in_channels, 4) / math.sqrt(4 * in_channels / 2)
        self.weight = torch.nn.Parameter(weight)
        self.stride = stride
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        weight = torch.zeros(self.out_channels, 4, self.in_channels, 4).to(x.device)
        weight[::, 0, ...] = self.weight
        weight[::, 1, ...] = self.weight[..., [3, 0, 1, 2]]
        weight[::, 2, ...] = self.weight[..., [2, 3, 0, 1]]
        weight[::, 3, ...] = self.weight[..., [1, 2, 3, 0]]
        x = torch.nn.functional.conv2d(x, weight.reshape(self.out_channels * 4, self.in_channels * 4, 1, 1), stride=1,
                                       padding=0)
        if (self.stride != 1):
            x = self.pool(x)
        return x


class C_4_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, stride=1):
        super(C_4_3x3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight = torch.randn(out_channels, in_channels, 4, 3, 3) / math.sqrt(4 * in_channels * 9 / 2)
        self.weight = torch.nn.Parameter(weight)
        self.pool = nn.MaxPool2d(2, 2)
        self.stride = stride
        self.bias = None
        if (bias):
            self.bias = torch.nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        weight = torch.zeros(self.out_channels, 4, self.in_channels, 4, 3, 3).to(x.device)
        weight[::, 0, ...] = self.weight
        weight[::, 1, ...] = torch.rot90(self.weight[..., [3, 0, 1, 2], ::, ::], 1, [3, 4])
        weight[::, 2, ...] = torch.rot90(self.weight[..., [2, 3, 0, 1], ::, ::], 2, [3, 4])
        weight[::, 3, ...] = torch.rot90(self.weight[..., [1, 2, 3, 0], ::, ::], 3, [3, 4])
        x = torch.nn.functional.conv2d(x, weight.reshape(self.out_channels * 4, self.in_channels * 4, 3, 3), padding=1)
        if (self.stride != 1):
            x = self.pool(x)
        if (self.bias is not None):
            b, c, w, h = x.shape
            x = (x.reshape(b, c // 4, 4, h, w) + self.bias.reshape(1, -1, 1, 1, 1)).reshape(x.shape)
        return x


class C_4_BN(nn.Module):
    def __init__(self, in_channels):
        super(C_4_BN, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        return self.bn(x.reshape(b, c // 4, 4, h, w)).reshape(x.size())


class D_4_BN(nn.Module):
    def __init__(self, in_channels, momentum=0.1):
        super(D_4_BN, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels, momentum=momentum)

    def forward(self, x):
        b, c, h, w = x.shape
        return self.bn(x.reshape(b, c // 8, 8, h, w)).reshape(x.size())


class C_4_1x1_(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C_4_1x1_, self).__init__()
        self.net = nn.Conv3d(in_channels, out_channels, 1, bias=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.net(x.view(b, c // 4, 4, h, w)).reshape(b, self.out_channels * 4, h, w)
        return x


class E4_C4(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 reduction_ratio=2,
                 groups=1,
                 stride=1
                 ):
        super(E4_C4, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.group_channels = groups
        self.groups = self.out_channels // self.group_channels
        self.dim_g = 4

        self.v = nn.Sequential(C_4_1x1(in_channels, out_channels))

        self.conv1 = nn.Sequential(C_4_1x1(in_channels, int(in_channels // reduction_ratio)),
                                    nn.GroupNorm(int(in_channels // reduction_ratio),int(in_channels // reduction_ratio)*4),
                                    nn.ReLU()
                                    )
        self.conv2 = nn.Sequential(C_4_1x1_(int(in_channels // reduction_ratio), kernel_size ** 2 * self.groups),
                                   )

        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, 4, h, w)
        weight[::, ::, ::, ::, 1, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 1, ::, ::], 1, [2, 3])
        weight[::, ::, ::, ::, 2, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 2, ::, ::], 2, [2, 3])
        weight[::, ::, ::, ::, 3, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 3, ::, ::], 3, [2, 3])
        weight = weight.reshape(b, self.groups, self.kernel_size ** 2, 4, h, w).unsqueeze(2).transpose(3, 4)

        x = self.v(x)
        out = self.unfold(x).view(b, self.groups, self.group_channels, 4, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=4).view(b, self.out_channels * 4, h, w)
        return out



class D_4_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(D_4_1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight = torch.randn(out_channels, in_channels, 8) / math.sqrt(8 * in_channels / 2)
        self.weight = torch.nn.Parameter(weight)
        self.stride = stride
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        weight = torch.zeros(self.out_channels, 8, self.in_channels, 8).to(x.device)
        weight[::, 0, ...] = self.weight
        weight[::, 1, ...] = self.weight[..., [3, 0, 1, 2, 5, 6, 7, 4]]
        weight[::, 2, ...] = self.weight[..., [2, 3, 0, 1, 6, 7, 4, 5]]
        weight[::, 3, ...] = self.weight[..., [1, 2, 3, 0, 7, 4, 5, 6]]
        weight[::, 4, ...] = self.weight[..., [4, 5, 6, 7, 0, 1, 2, 3]]
        weight[::, 5, ...] = self.weight[..., [5, 6, 7, 4, 3, 0, 1, 2]]
        weight[::, 6, ...] = self.weight[..., [6, 7, 4, 5, 2, 3, 0, 1]]
        weight[::, 7, ...] = self.weight[..., [7, 4, 5, 6, 1, 2, 3, 0]]
        x = torch.nn.functional.conv2d(x, weight.reshape(self.out_channels * 8, self.in_channels * 8, 1, 1), stride=1,
                                       padding=0)
        if (self.stride != 1):
            x = self.pool(x)
        return x


class D_4_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, stride=1):
        super(D_4_3x3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight = torch.randn(out_channels, in_channels, 8, 3, 3) / math.sqrt(8 * in_channels * 9 / 2)
        self.weight = torch.nn.Parameter(weight)
        self.pool = nn.MaxPool2d(2, 2)
        self.stride = stride
        self.bias = None
        if (bias):
            self.bias = torch.nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        weight = torch.zeros(self.out_channels, 8, self.in_channels, 8, 3, 3).to(x.device)
        weight[::, 0, ...] = self.weight
        weight[::, 1, ...] = torch.rot90(self.weight[..., [3, 0, 1, 2, 5, 6, 7, 4], ::, ::], 1, [3, 4])
        weight[::, 2, ...] = torch.rot90(self.weight[..., [2, 3, 0, 1, 6, 7, 4, 5], ::, ::], 2, [3, 4])
        weight[::, 3, ...] = torch.rot90(self.weight[..., [1, 2, 3, 0, 7, 4, 5, 6], ::, ::], 3, [3, 4])
        weight[::, 4, ...] = torch.rot90(self.weight[..., [4, 5, 6, 7, 0, 1, 2, 3], ::, ::].transpose(3, 4), 3, [3, 4])
        weight[::, 5, ...] = torch.rot90(self.weight[..., [5, 6, 7, 4, 3, 0, 1, 2], ::, ::].transpose(3, 4), 2, [3, 4])
        weight[::, 6, ...] = torch.rot90(self.weight[..., [6, 7, 4, 5, 2, 3, 0, 1], ::, ::].transpose(3, 4), 1, [3, 4])
        weight[::, 7, ...] = torch.rot90(self.weight[..., [7, 4, 5, 6, 1, 2, 3, 0], ::, ::].transpose(3, 4), 0, [3, 4])

        x = torch.nn.functional.conv2d(x, weight.reshape(self.out_channels * 8, self.in_channels * 8, 3, 3), padding=1)
        if (self.stride != 1):
            x = self.pool(x)
        if (self.bias is not None):
            b, c, w, h = x.shape
            x = (x.reshape(b, c // 8, 8, h, w) + self.bias.reshape(1, -1, 1, 1, 1)).reshape(x.shape)
        return x


class D_4_1x1_(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(D_4_1x1_, self).__init__()
        self.net = nn.Conv3d(in_channels, out_channels, 1, bias=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.net(x.view(b, c // 8, 8, h, w)).reshape(b, self.out_channels * 8, h, w)
        return x


class E4_D4(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 reduction_ratio=2,
                 groups=1,
                 stride=1):
        super(E4_D4, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.group_channels = groups
        self.groups = self.out_channels // self.group_channels
        self.dim_g = 8

        self.v = nn.Sequential(D_4_1x1(in_channels, out_channels))

        self.conv1 = nn.Sequential(D_4_1x1(in_channels, int(in_channels // reduction_ratio)),
                                   nn.GroupNorm(int(in_channels // reduction_ratio), int(in_channels // reduction_ratio)*8),
                                   nn.ReLU()
                                   )
        self.conv2 = nn.Sequential(D_4_1x1_(int(in_channels // reduction_ratio), kernel_size ** 2 * self.groups),
                                   )
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, 8, h, w)
        weight[::, ::, ::, ::, 1, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 1, ::, ::], 1, [2, 3])
        weight[::, ::, ::, ::, 2, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 2, ::, ::], 2, [2, 3])
        weight[::, ::, ::, ::, 3, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 3, ::, ::], 3, [2, 3])
        weight[::, ::, ::, ::, 4, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 4, ::, ::].transpose(2, 3), 3, [2, 3])
        weight[::, ::, ::, ::, 5, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 5, ::, ::].transpose(2, 3), 2, [2, 3])
        weight[::, ::, ::, ::, 6, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 6, ::, ::].transpose(2, 3), 1, [2, 3])
        weight[::, ::, ::, ::, 7, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 7, ::, ::].transpose(2, 3), 0, [2, 3])
        weight = weight.view(b, self.groups, self.kernel_size ** 2, 8, h, w).unsqueeze(2).transpose(3, 4)
        x = self.v(x)
        out = self.unfold(x).view(b, self.groups, self.group_channels, 8, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=4).view(b, self.out_channels * 8, h, w)
        return out


def C_4_rot(x):
    b, c, h, w = x.shape
    return torch.rot90(x.view(b, c // 4, 4, h, w)[::, ::, [3, 0, 1, 2]], 1, [3, 4]).reshape(x.shape)

def D_4_rot(x):
    b, c, h, w = x.shape
    return torch.rot90(x.view(b, c // 8, 8, h, w)[::, ::, [3, 0, 1, 2, 5, 6, 7, 4]], 1, [3, 4]).reshape(x.shape)


def D_4_m(x):
    b, c, h, w = x.shape
    return torch.rot90(x.view(b, c // 8, 8, h, w)[::, ::, [4, 5, 6, 7, 0, 1, 2, 3]].transpose(3, 4), 3, [3, 4]).reshape(
        x.shape)


class C_4_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C_4_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight = torch.randn(out_channels, in_channels, 3, 3) / math.sqrt(9 * in_channels / 2)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, input):
        weight = torch.zeros(self.out_channels, 4, self.in_channels, 3, 3).to(input.device)
        weight[::, 0] = self.weight
        weight[::, 1] = torch.rot90(self.weight[::], 1, [2, 3])
        weight[::, 2] = torch.rot90(self.weight[::], 2, [2, 3])
        weight[::, 3] = torch.rot90(self.weight[::], 3, [2, 3])
        out = nn.functional.conv2d(input, weight.reshape(self.out_channels * 4, self.in_channels, 3, 3), padding=1)
        return out


class D_4_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(D_4_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight = torch.randn(out_channels, in_channels, 3, 3) / math.sqrt(9 * in_channels / 2)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, input):
        weight = torch.zeros(self.out_channels, 8, self.in_channels, 3, 3).to(input.device)
        weight[::, 0] = self.weight
        weight[::, 1] = torch.rot90(self.weight[::], 1, [2, 3])
        weight[::, 2] = torch.rot90(self.weight[::], 2, [2, 3])
        weight[::, 3] = torch.rot90(self.weight[::], 3, [2, 3])
        weight[::, 4] = torch.rot90(self.weight[::].transpose(2, 3), 3, [2, 3])
        weight[::, 5] = torch.rot90(self.weight[::].transpose(2, 3), 2, [2, 3])
        weight[::, 6] = torch.rot90(self.weight[::].transpose(2, 3), 1, [2, 3])
        weight[::, 7] = torch.rot90(self.weight[::].transpose(2, 3), 0, [2, 3])
        out = nn.functional.conv2d(input, weight.reshape(self.out_channels * 8, self.in_channels, 3, 3), padding=1)
        return out


class C_4_Pool(nn.Module):
    def __init__(self):
        super(C_4_Pool, self).__init__()
        self.pool = nn.MaxPool3d((4, 1, 1), (4, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        return self.pool(x.reshape(b, c // 4, 4, h, w)).squeeze(2)


class D_4_Pool(nn.Module):
    def __init__(self):
        super(D_4_Pool, self).__init__()
        self.pool = nn.MaxPool3d((8, 1, 1), (8, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        return self.pool(x.reshape(b, c // 8, 8, h, w)).squeeze(2)




#Right

