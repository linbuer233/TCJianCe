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

#继承nn.Module类，构建网络模型
class LogicNet(nn.Module):
    def __init__(self,inputdim,hiddendim,outputdim):#初始化网络结构
        super(LogicNet,self).__init__()
        self.Linear1 = nn.Linear(inputdim,hiddendim) #定义全连接层
        self.Linear2 = nn.Linear(hiddendim,outputdim)#定义全连接层
        self.criterion = nn.L1Loss() #定义交叉熵函数

    def forward(self,x): #搭建用两层全连接组成的网络模型
        x = self.Linear1(x)#将输入数据传入第1层
        x = torch.tanh(x)#对第一层的结果进行非线性变换
        x = self.Linear2(x)#再将数据传入第2层
#        print("LogicNet")
        return x

    def predict(self,x):#实现LogicNet类的预测接口
        #调用自身网络模型，并对结果进行softmax处理,分别得出预测数据属于每一类的概率
        pred = self.forward(x)
        return pred  #返回每组预测概率中最大的索引

    def getloss(self,x,y): #实现LogicNet类的损失值计算接口
        y_pred = self.forward(x)
        loss = self.criterion(y_pred,y)#计算损失值得交叉熵
        return loss


hidden_dim = 100
num_layers = 1

network = LogicNet(2, hidden_dim, 2)

# 指定设备
device = torch.device("cpu")
network.to(device)

x = torch.rand(1000, 1, 2)
a = torch.rand(1, 1, 2)
y = 2 * x
print(y[2, :, :])
optimizer = torch.optim.Adam(network.parameters(), lr=0.005)

print(network(a).shape)
lossal = []
for j in range(1000):
        loss = network.getloss(x[j, :, :], y[j, :, :])
        lossal.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print(network.predict(x[2, :, :]))
print(len(lossal))

import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.arange(0, len(lossal), 1), lossal)
plt.show()
