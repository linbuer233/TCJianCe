import numpy as np
import pandas as pd

year = [2017, 2018, 2020]


def speed(lon1, lon2, lat1, lat2):
    R = 6371
    wa = lat1 * np.pi / 180
    wb = lat2 * np.pi / 180
    ja = lon1 * np.pi / 180
    jb = lon2 * np.pi / 180
    return np.round(R * np.arccos(np.sin(wa) * np.sin(wb) + np.cos(wa) * np.cos(wb) * np.cos(ja - jb)), 2)


def jiao(lon1, lon2, lat1, lat2):
    return np.round(np.arctan((lat2 - lat1) / (lon2 - lon1)), 2)


# 2017073106
# 把时间间隔处理为6小时
for year_i in year:
    print(year_i)
    df1 = pd.read_csv(r'../data/CMATC7-8/CH' + str(year_i) + '.csv')
    df1 = df1.reset_index(drop=True)
    print(df1)
    df = df1
    for i in range(len(df1)):
        print(i)
        # print(df.loc[i])
        if df1['时间'].loc[i] == 66666:
            continue
        if str(df1['时间'].loc[i])[-2:] in (['03', '09', '15', '21']):
            df = df.drop(i, axis=0)
            continue
        if str(df1['时间'].loc[i])[6:8] in (['31']):
            df = df.drop(i, axis=0)
    df = df.reset_index(drop=True)
    df['速度'] = np.zeros(len(df))
    df['角度'] = np.zeros(len(df))
    for i in range(1, len(df), 1):
        # if i == 0:
        #     continue
        # if (df.loc[i - 1]['时间'] == 66666) or (df.loc[i]['时间'] == 66666):
        #     continue
        # df.loc[i, ['速度']] = [speed(df.loc[i - 1]['经度'], df.loc[i]['经度'], df.loc[i - 1]['纬度'], df.loc[i]['纬度'])]
        # df.loc[i, ['角度']] = [jiao(df.loc[i - 1]['经度'], df.loc[i]['经度'], df.loc[i - 1]['纬度'], df.loc[i]['纬度'])]
        if i == len(df) - 1:
            break
        if (df.loc[i + 1]['时间'] == 66666) or (df.loc[i]['时间'] == 66666):
            continue
        df.loc[i, ['速度']] = [speed(df.loc[i]['经度'], df.loc[i + 1]['经度'], df.loc[i]['纬度'], df.loc[i + 1]['纬度'])]
        df.loc[i, ['角度']] = [jiao(df.loc[i]['经度'], df.loc[i + 1]['经度'], df.loc[i]['纬度'], df.loc[i + 1]['纬度'])]
    df.to_csv(r'../data/2017-2020_7-8/CH' + str(year_i) + '.csv')
