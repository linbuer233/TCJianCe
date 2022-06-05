# 模型部分
import torch
from torch import nn


# ds = xr.open_dataset('../data/all.nc')
# print(ds)


# class Connet(nn.Module):
#
#     def __init__(self):
#         super(Connet, self).__init__()
#         #
#         self.conv1 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
#         self.conv2 = nn.Conv2d(in_channels=10, out_channels=12, kernel_size=3)
#         self.conv3 = nn.Conv2d(in_channels=12, out_channels=10, kernel_size=3)
#
#
#     def forward(self, t):
#         #
#         t = self.conv1(t)
#         print("1juan:",t.shape)
#         t = F.relu(t)
#         t = F.max_pool2d(t, kernel_size=3, stride=3)
#         print("1pool:", t.shape)
#         #
#         t = self.conv2(t)
#         print("2juan:", t.shape)
#         t = F.relu(t)
#         t = F.max_pool2d(t, kernel_size=3, stride=3)
#         print("2pool:", t.shape)
#
#         #
#         t = self.conv3(t)
#         print("3juan:", t.shape)
#         t = F.avg_pool2d(t, kernel_size=3, stride=4)
#         print("3pool:", t.shape)
#
#         return t#.reshape(t.shape[:2])

class GRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size, num_layers):
        super(GRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers, bidirectional=True)

        self.Linear1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, output_size)

        self.criterion = nn.L1Loss()  # 定义损失函数

    def forward(self, t):
        t, _ = self.gru(t)
        t = self.Linear1(t)
        # t = torch.tanh(t)
        output = self.Linear2(t)

        return output

    def init_zeros_state(self):
        init_hidden = torch.zeros(self.num_layers * 2, self.hidden_dim).to(device)
        return init_hidden

    def predict(self, x):
        pred = self.forward(x)
        return pred

    def getloss(self, x, y):
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)  # 交叉熵计算误差
        return loss


hidden_dim = 10
num_layers = 1

network = GRNN(2, hidden_dim, 2, num_layers)

# 指定设备
device = torch.device("cpu")
network.to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=0.005)


x = torch.rand(100, 1, 2)
y = 2 * x
print(y[2, :, :])


# print(network(a))
lossal = []
for i in range(50):
    for j in range(100):
        loss = network.getloss(x[j, :, :], y[j, :, :])
        lossal.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # optimizer.zero_grad()
    # # loss.backward()
    # optimizer.step()

print(network.predict(x[2, :, :]))
print(len(lossal))

import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.arange(0, len(lossal), 1), lossal)
plt.show()
