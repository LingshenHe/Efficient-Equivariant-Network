import torch
import torch.nn as nn
from .group_op import E4_C4, C_4_Conv, C_4_BN, C_4_Pool




class E4_net(nn.Module):
    def __init__(self, kernel_size=5, groups=8, reduction_ratio=1, drop=0.2):
        super(E4_net, self).__init__()
        self.conv1=C_4_Conv(1, 16)
        self.conv2=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.conv3=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.conv4=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.conv5=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.conv6=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.conv7=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.pool=nn.MaxPool2d(2,2)

        self.bn1=C_4_BN(16)
        self.bn2=C_4_BN(16)
        self.bn3=C_4_BN(16)
        self.bn4=C_4_BN(16)
        self.bn5=C_4_BN(16)
        self.bn6=C_4_BN(16)
        self.bn7=C_4_BN(16)

        self.drop=nn.Dropout(drop)


        self.group_pool=C_4_Pool()
        self.global_pool=nn.AdaptiveMaxPool2d(1)

        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(16, 10),
        )


    def forward(self, x):
        x=torch.relu(self.bn1(self.conv1(x)))
        x=torch.relu(self.bn2(self.conv2(x)))
        x=self.pool(x)
        x=self.drop(torch.relu(self.bn3(self.conv3(x))))
        x=self.drop(torch.relu(self.bn4(self.conv4(x))))
        x=self.pool(x)
        x=self.drop(torch.relu(self.bn5(self.conv5(x))))
        x=self.drop(torch.relu(self.bn6(self.conv6(x))))
        x=self.drop(torch.relu(self.bn7(self.conv7(x))))
        x=self.group_pool(x)
        x=self.global_pool(x).reshape(x.size(0),-1)
        x=self.fully_net(x)
        return x




