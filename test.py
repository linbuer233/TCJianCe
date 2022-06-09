import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

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
    hidden_dim = [8, 6, 1]  # 隐藏层纬度，后者是输出的纬度
    kernel_size = (3, 3)  # kernel size for two stacked hidden layer
    num_layers = 3  # number of stacked hidden layer
    model = convGRU.ConvGRU(input_size=(height, width),
                            input_dim=channels,
                            hidden_dim=hidden_dim,
                            kernel_size=kernel_size,
                            num_layers=num_layers,
                            dtype=dtype,
                            batch_first=True,
                            bias=True,
                            return_all_layers=False)

    # 加载训练好的模型
    PATH = './taifennet.pth'
    model.load_state_dict(torch.load(PATH))
    # 指定设备
    device = torch.device("cpu")
    model.to(device)
    # 读取测试数据
    c2020 = name('../data/2017-2020_7-8/CH2017.csv')
    # c2020 = name('../data/2017-2020_7-8/CH2020.csv')
    # --------------------------测试阶段------------------------------------#
    testds = xr.open_dataset('../data/x_train.nc')
    # testds = xr.open_dataset('../data/x_test.nc')

    lossal=[]
    er=[]
    for i in c2020:
        # -------------------ylabel---------------------#
        df = pd.read_csv('../data/2017-2020_7-8/CH2017.csv')
        # df = pd.read_csv('../data/2017-2020_7-8/CH2020.csv')
        #
        ytest = df[df['名字'] == i]
        ytest = ytest.reset_index(drop=True).loc[1:len(ytest) - 2]

        # 对CMA进行剔除
        ytest = ytest[jiaoji(list((ytest['经度'] > 103).values), list((ytest['经度'] < 177).values))]
        ytest = ytest[jiaoji(list((ytest['纬度'] > -7).values), list((ytest['纬度'] < 47).values))]
        ytest = ytest[ytest['强度'] > 1]
        print("------------------------")
        # 读取时间信息方便切割数据集
        timelist = ytest['时间'].reset_index(drop=True).values
        # 读取中心经纬读
        lat = ytest['纬度'].reset_index(drop=True).values
        lon = ytest['经度'].reset_index(drop=True).values
        lat1 = lat[0]
        lon1 = lon[0]
        """
        通过第一个点的位置框选数据，检测下一时刻的台风位置，得到位置后，依据该位置再次框选数据，预测下下时刻的数据，以此类推
        """

        for t_i in range(1, len(ytest), 1):
            x_test = np.zeros((1, 1, channels, height, width))
            timedate = str(timelist[t_i])[0:4] + '-' + str(timelist[t_i])[4:6] + '-' + str(timelist[t_i])[
                                                                                       6:8] + 'T' + str(
                timelist[t_i])[8:10]
            # 把经纬度的小数位设为 .25 整数倍
            latst = lat_lon(lat1) + 2.5
            latend = lat_lon(lat1) - 2.5
            lonst = lat_lon(lon1) - 2.5
            lonend = lat_lon(lon1) + 2.5

            # ------------------------------测试数据------------------------#

            # 850hPa涡度
            x_test[0, 0, 0, :, :] = (testds['vo'].loc[timedate,
                                     latst:latend, lonst:lonend].data - np.nanmean(
                testds['vo'].loc[timedate,
                latst:latend,
                lonst:lonend].data)) / np.nanstd(testds[
                                                     'vo'].loc[
                                                 timedate,
                                                 latst:latend,
                                                 lonst:lonend].data)

            # 海平面气压
            x_test[0, 0, 1, :, :] = (testds['sp'].loc[timedate,
                                     latst:latend, lonst:lonend].data - np.nanmean(
                testds['sp'].loc[timedate,
                latst:latend,
                lonst:lonend].data)) / np.nanstd(testds[
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
            x_test[0, 0, 2:4, :, :] = (testds['u'].loc[timedate, [200, 850], latst:latend,
                                       lonst:lonend].data - np.nanmean(testds['u'].loc[timedate, [200, 850],
                                                                       latst:latend, lonst:lonend].data,
                                                                       axis=(1, 2)).reshape((2, 1, 1))) / \
                                      np.nanstd(testds['u'].loc[timedate, [200, 850],
                                                latst:latend,
                                                lonst:lonend].data,
                                                axis=(1, 2)).reshape((2, 1, 1))
            # V
            x_test[0, 0, 4:6, :, :] = (testds['v'].loc[timedate, [200, 850],
                                       latst:latend, lonst:lonend].data - np.nanmean(
                testds['v'].loc[timedate,
                [200, 850],
                latst:latend,
                lonst:lonend].data, axis=(1, 2)).reshape((2, 1, 1))) / np.nanstd(
                testds['v'].loc[timedate, [200, 850],
                latst:latend, lonst:lonend].data,
                axis=(1, 2)).reshape((
                2, 1,
                1))
            # -----------------------开始测试--------------------------#
            x_test = np.array(x_test)
            x_test = torch.Tensor(x_test)

            output, _ = model(x_test)
            output[0] = torch.where(output[0] > 0, 1, 0)
            # 根据output矩阵计算经纬度
            output = output[0].numpy().reshape((21, 21))

            # 选择为一的位置，计算出经纬度
            b = np.where(output == 1)
            #
            latlist = b[0]
            lonlist = b[1]
            # print(latlist, lonlist)
            templat = templon = 0
            print(timelist[t_i],i)
            for lat_i, lon_i in zip(latlist, lonlist):
                templat += lat1 - 2.5 + 0.25 * lat_i
                templon += lon1 - 2.5 + 0.25 * lon_i
            try:
                lat1=templat/len(latlist)
                lon1=templon/len(lonlist)
                loss = ((lat1 - lat[t_i]) + (lon1 - lon[t_i])) / 2
                lossal.append(loss)
            except:
                er.append((timelist[t_i],i))
                lossal.append(-100)

            #
    print(lossal)
    plt.plot(range(len(lossal)),lossal)
    plt.show()