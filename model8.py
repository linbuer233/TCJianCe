import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import nn
from torch.nn import functional as F


class Connet(nn.Module):

    def __init__(self):
        super(Connet, self).__init__()
        #
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=4, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=2)
        self.Linear1 = nn.Linear(in_features=2, out_features=10)
        self.Linear2 = nn.Linear(in_features=10, out_features=2)
        self.criterion = nn.MSELoss(reduction='sum')

    def forward(self, t):
        #
        t = self.conv1(t)
        # print("1juan:", t.shape)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=4, stride=1)
        # print("1pool:", t.shape)
        #
        t = self.conv2(t)
        # print("2juan:", t.shape)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=2)
        # print("2pool:", t.shape)

        #
        t = self.conv3(t)
        # print("3juan:", t.shape)
        # t = F.relu(t)
        t = F.avg_pool2d(t, kernel_size=2, stride=1)
        # print("3pool:", t.shape)

        #
        t = self.conv4(t)
        # t = F.relu(t)
        # print("4juan:", t.shape)
        t = F.avg_pool2d(t, kernel_size=2, stride=2)
        # print("4pool:", t.shape)
        t = t.reshape(1, 1, 2)
        t = self.Linear1(t)
        t = torch.tanh(t)
        t = self.Linear2(t)
        return t  # .reshape(t.shape[:2])

    def predict(self, x):
        pred = self.forward(x)
        return pred

    def getloss(self, x, y):
        y_pred = self.forward(x)
        y_pred = y_pred.reshape(1, 1, 2)
        loss = self.criterion(y_pred, y)  # MSE计算误差
        return loss



if __name__ == '__main__':
    network = Connet()

    # 指定设备
    device = torch.device("cpu")
    network.to(device)

    a = torch.rand(6, 21, 21)
    print(network(a))

    device = torch.device("cpu")
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0005)

    x = torch.rand(10, 10, 21, 21)
    y = torch.rand(10, 1, 1, 2)
    print(y[2, :, :, :])

    lossal = []
    for i in range(400):
        for j in range(10):
            loss = network.getloss(x[j, :, :, :], y[j, :, :, :])
            lossal.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(network.predict(x[2, :, :, :]))
    print(len(lossal))

    import matplotlib.pyplot as plt
    import numpy as np

    plt.plot(np.arange(0, len(lossal), 1), lossal)
    plt.show()

