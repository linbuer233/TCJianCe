# from torch.autograd import Variable
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import nn

# 导入ConvGRU
import convGRU

ts = time.time()


# 手动实现交集
def jiaoji(list1, list2):
    TFlist = []
    for i in range(len(list1)):
        if (list1[i] == list2[i]) and (list2[i] == True):
            TFlist.append(True)
        else:
            TFlist.append(False)
    return TFlist


# 通过台风名字选取台风的信息（排除nameless）
def name(path):
    df = pd.read_csv(path)
    name = df['名字']
    a = {}
    for i in name:
        if i not in a:
            a[i] = 1
        else:
            a[i] += 1
    dell = []
    for i in a.items():
        if i[1] <= 8 or i[0] == '(nameless)':
            dell.append(i[0])
    for i in dell:
        del a[i]
    Name = []
    for i in a.keys():
        Name.append(i)
    return Name


# 处理台风经纬度，使其小数位是0，和 .25 的整数倍
def lat_lon(lat):
    a = round(lat, 0)
    if ((lat - a) == 0) or ((lat - 0) <= 0.125):  # .0
        return a
    if ((lat - a) > 0.125) or ((lat - a) <= 0.375):  # .25
        return a + 0.25
    if ((lat - a) > 0.375) or ((lat - a) <= 0.625):  # .50
        return a + 0.5
    if ((lat - a) > 0.625) or ((lat - a) <= 0.875):  # .75
        return a + 0.75
    if (lat - a) > 0.875:  # .0
        return a + 1


if __name__ == '__main__':
    dtype = torch.FloatTensor
    height = width = 21
    channels = 6  # 通道数
    hidden_dim = [4, 2]  # 隐藏层纬度，后者是输出的纬度
    kernel_size = (3, 3)  # kernel size for two stacked hidden layer
    num_layers = 2  # number of stacked hidden layer
    model = convGRU.ConvGRU(input_size=(height, width),
                            input_dim=channels,
                            hidden_dim=hidden_dim,
                            kernel_size=kernel_size,
                            num_layers=num_layers,
                            dtype=dtype,
                            batch_first=True,
                            bias=True,
                            return_all_layers=False)

    # 确定优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.01)
    # 定义损失函数
    criterion = nn.MSELoss(reduction='mean')
    # 训练次数
    ethches = 2
    # 读取训练集
    trainds = xr.open_dataset('../data/x_train.nc')
    print(trainds)
    #
    batch_size = 1  # 样本批次
    time_steps = 100  # 时间步长
    c2017 = name('../data/2017-2020_7-8/CH2017.csv')
    c2018 = name('../data/2017-2020_7-8/CH2018.csv')
    c2019 = name('../data/2017-2020_7-8/CH2019.csv')
    c2020 = name('../data/2017-2020_7-8/CH2020.csv')
    trainname = [c2017, c2018, c2019]
    # print(len(c2017),len(c2018)
    # 指定设备
    device = torch.device("cpu")
    model.to(device)

    # 训练阶段
    lossal = []
    for i in range(ethches):
        for name_i, name_j in zip(range(2), ['2017', '2018']):
            for j in trainname[name_i]:
                """
                一次训练一个台风序列，利用台风名字选取路径数据，训练100次
                """
                # -------------------ylabel---------------------#
                df = pd.read_csv('../data/2017-2020_7-8/CH' + name_j + '.csv')
                #
                #
                ytrain = df[df['名字'] == j]
                ytrain = ytrain.reset_index(drop=True).loc[1:len(ytrain) - 2]

                # 对CMA进行剔除
                ytrain = ytrain[jiaoji(list((ytrain['经度'] > 103).values), list((ytrain['经度'] < 177).values))]
                ytrain = ytrain[jiaoji(list((ytrain['纬度'] > -7).values), list((ytrain['纬度'] < 47).values))]
                ytrain = ytrain[ytrain['强度'] > 1]
                print("------------------------")
                # 读取时间信息方便切割数据集
                timelist = ytrain['时间'].reset_index(drop=True).values
                # 读取中心经纬读
                lat = ytrain['纬度'].reset_index(drop=True).values
                lon = ytrain['经度'].reset_index(drop=True).values
                # 风速和角度
                ylabel = np.zeros((len(ytrain), 2))
                ylabel[:, 0] = lat   # ytrain['速度']
                ylabel[:, 1] = lon   # ytrain['角度']
                # ylabel = torch.rand(len(ytrain), 2)
                # ------------------训练集-----------------------#
                """
                通过台风中心点选取周围的环境场，分辨率0.25°，故选取21*21的方格的数据，范围5°*5° 500km*500km左右
                ytrain的长度就是 <时间长度> 
                每个时刻的中心点不一样，所以嵌套循环
                """
                x_train = np.zeros((batch_size, len(ytrain), channels, 21, 21))
                for t_i in range(len(ytrain)):
                    timedate = str(timelist[t_i])[0:4] + '-' + str(timelist[t_i])[4:6] + '-' + str(timelist[t_i])[
                                                                                               6:8] + 'T' + str(
                        timelist[t_i])[8:10]
                    # 把经纬度的小数位设为 .25 整数倍
                    latst = lat_lon(lat[t_i]) + 2.5
                    latend = lat_lon(lat[t_i]) - 2.5
                    lonst = lat_lon(lon[t_i]) - 2.5
                    lonend = lat_lon(lon[t_i]) + 2.5
                    # 850hPa涡度
                    # 850hPa涡度
                    x_train[0, t_i, 0, :, :] = (trainds['vo'].loc[timedate,
                                                latst:latend, lonst:lonend].data - np.nanmean(
                        trainds['vo'].loc[timedate,
                        latst:latend,
                        lonst:lonend].data)) / np.nanstd(trainds[
                                                             'vo'].loc[
                                                         timedate,
                                                         latst:latend,
                                                         lonst:lonend].data)

                    # 海平面气压
                    x_train[0, t_i, 1, :, :] = (trainds['sp'].loc[timedate,
                                                latst:latend, lonst:lonend].data - np.nanmean(
                        trainds['sp'].loc[timedate,
                        latst:latend,
                        lonst:lonend].data)) / np.nanstd(trainds[
                                                             'sp'].loc[
                                                         timedate,
                                                         latst:latend,
                                                         lonst:lonend].data)
                    # 海温
                    # x_train[0, t_i, 2, :, :] = trainds['sst'].loc[timedate,
                    #                            latst:latend, lonst:lonend].data
                    # 温度
                    # x_train[0, t_i, 2:5, :, :] = trainds['t'].loc[timedate, 500:,
                    #                              latst:latend, lonst:lonend].data
                    # U
                    x_train[0, t_i, 2:4, :, :] = (trainds['u'].loc[timedate, [200, 850], latst:latend,
                                                  lonst:lonend].data - np.nanmean(trainds['u'].loc[timedate, [200, 850],
                                                                                  latst:latend, lonst:lonend].data,
                                                                                  axis=(1, 2)).reshape(2, 1, 1)) / \
                                                 np.nanstd(trainds['u'].loc[timedate, [200, 850],
                                                           latst:latend,
                                                           lonst:lonend].data,
                                                           axis=(1, 2)).reshape(2, 1, 1)
                    # V
                    x_train[0, t_i, 4:6, :, :] = (trainds['v'].loc[timedate, [200, 850],
                                                  latst:latend, lonst:lonend].data - np.nanmean(
                        trainds['v'].loc[timedate,
                        [200, 850],
                        latst:latend,
                        lonst:lonend].data, axis=(1, 2)).reshape(2, 1, 1)) / np.nanstd(
                        trainds['v'].loc[timedate, [200, 850],
                        latst:latend, lonst:lonend].data,
                        axis=(1, 2)).reshape(
                        2, 1,
                        1)
                    # x_train = np.where(x_train != np.nan, x_train, 0)
                x_train = torch.rand((batch_size, len(ytrain), channels, 21, 21))
                # print(x_train[0, -1, 0, :, :])
                # a = input()
                # ----------------------------开始训练---------------------------------#
                # 设置基本参数
                batch_size = 1  # 样本批次
                time_steps = len(ytrain)  # 时间步长
                # 把 xtrain和ylabel 转化为 tensor
                # x_train = np.array(x_train)
                # x_train = torch.Tensor(x_train)
                ylabel = torch.Tensor(ylabel)
                # lossal
                hid = model._init_hidden(batch_size)

                output, _ = model(x_train)
                y_pred = output[0].nanmean(axis=(3, 4)).reshape(len(ytrain), 2)
                loss = criterion(y_pred, ylabel)
                print("loss: ", loss.item())
                lossal.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=1.)
                optimizer.step()

                """
                # print(hid[0][0].shape)
                for time_i in range(len(ytrain)):
                    hid = model._init_hidden(batch_size)
                    output, hid = model(x_train[0, time_i, :, :, :].reshape(1, 1, channels, height, width),
                                        hidden_state=hid)
                    y_pred = output[0].nanmean(axis=(3, 4)).reshape(1, 2)
                    loss = criterion(y_pred, ylabel[time_i, :])
                    if np.isnan(loss.item()):
                        print(time_i, name_i, name_j, j)
                        a = input()
                    print("loss: ", loss.item())
                    lossal.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=1.)
                    optimizer.step()
                """
    print(len(lossal))
    plt.plot(np.arange(0, len(lossal), 1), lossal)
    plt.show()
    # print(output[0].shape)
    print("time:  ", time.time() - ts)
    a = input()
    # x_train=
    # print("time:  ", time.time() - ts)
