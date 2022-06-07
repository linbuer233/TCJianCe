import torch
from torch import nn
from torch.nn import functional as F


class CRnet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_size, num_layers):
        super(CRnet, self).__init__()
        # CNN部分
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=12, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=2, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=2)

        # RNN部分
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_size=2,
                          hidden_size=hidden_dim,
                          num_layers=num_layers, bidirectional=True)

        self.Linear1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, output_size)

        self.criterion = nn.MSELoss()

    def forward(self, t):
        # 一层卷积和池化
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=3)

        # 二层卷积和池化
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=3)

        # 三层卷积和池化
        t = self.conv3(t)
        t = F.avg_pool2d(t, kernel_size=3, stride=4)
        # print('t:  ', t.shape)
        # RNN
        t = t.reshape(1, 1, 2)
        t, _ = self.gru(t)
        t = self.Linear1(t)
        t = torch.tanh(t)
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
        loss = self.criterion(y_pred, y)  # MSE计算误差
        return loss


hidden_dim = 100
num_layers = 1

network = CRnet(10, hidden_dim, 2, num_layers)

# 指定设备
device = torch.device("cpu")
network.to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=0.005)

x = torch.rand(100, 10, 60, 60)
y = torch.rand(100, 1, 2)

print(y[2, :, :])
lossal = []
for i in range(50):
    for j in range(100):
        loss = network.getloss(x[j, :, :, :], y[j, :, :].reshape(1, 1, 2))
        lossal.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(parameters=network.parameters(),clip_value=1.)
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1, norm_type=2)
        optimizer.step()

print(network.predict(x[2, :, :, :]))
print(len(lossal))

import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.arange(0, len(lossal), 1), lossal)
plt.show()
