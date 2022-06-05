# 模型部分 
import torch
from torch import nn
from torch.nn import functional as F


# ds = xr.open_dataset('../data/all.nc')
# print(ds)


class Connet(nn.Module):

    def __init__(self):
        super(Connet, self).__init__()
        #
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=12, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=2, kernel_size=3)

    def forward(self, t):
        #
        t = self.conv1(t)
        print("1juan:", t.shape)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=3)
        print("1pool:", t.shape)
        #
        t = self.conv2(t)
        print("2juan:", t.shape)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=3)
        print("2pool:", t.shape)

        #
        t = self.conv3(t)
        print("3juan:", t.shape)
        t = F.avg_pool2d(t, kernel_size=3, stride=4)
        print("3pool:", t.shape)

        return t  # .reshape(t.shape[:2])


network = Connet()

# 指定设备
device = torch.device("cpu")
network.to(device)

a = torch.rand(10, 81, 81)
print(network(a).reshape(1,1,2))
