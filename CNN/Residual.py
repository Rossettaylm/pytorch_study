import torch
from torch import nn
from torch.nn.functional import relu


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)        # 输出特征图大小为(h/stride, w/stride)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride)    # 输出特征图大小为(h/stride, w/stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            # 将x调整到与y有相同的out_channels和特征图大小，当输出通道和输入通道不一致或stride不为1时，需要通过conv3来匹配x和y
            x = self.conv3(x)
        return relu(x+y)


residual_blk1 = Residual(3, 3)  # 输入通道为3，输出通道为3
x = torch.rand(4, 3, 6, 6)
y = residual_blk1(x)
print(y.shape)  # (4, 3, 6, 6)

residual_blk2 = Residual(3, 6, use_1x1conv=True, stride=2)
x = torch.rand(4, 3, 10, 6)
y = residual_blk2(x)
print(y.shape)  # (4, 6, 5, 3)
