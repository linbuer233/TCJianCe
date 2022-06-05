import pandas as pd 
import numpy as np 
year=[2017,2018,2019,2020]

def speed(lon1,lon2,lat1,lat2):
    R = 6371
    wa=lat1*np.pi/180
    wb=lat2*np.pi/180
    ja=lon1*np.pi/180
    jb=lon2*np.pi/180
    return R*np.arccos(np.sin(wa)*np.sin(wb)+np.cos(wa)*np.cos(wb)*np.cos(ja-jb))
def jiao(lon1,lon2,lat1,lat2):
    return np.arctan((lat2-lat1)/(lon2-lon1))

for year_i in year:
    df=pd.read_csv(r'../data/CMATC7-8/CH'+str(year_i)+'.csv')

    df['速度'] = np.zeros(len(df))
    df['角度'] = np.zeros(len(df))
    for i in range(1, len(df), 1):
        if i == 0:
            continue
        if (df.loc[i - 1]['时间'] == 66666) or (df.loc[i]['时间'] == 66666):
            continue
        df.loc[i, ['速度']] = [speed(df.loc[i - 1]['经度'], df.loc[i]['经度'], df.loc[i - 1]['纬度'], df.loc[i - 1]['纬度'])]
        df.loc[i, ['角度']] = [jiao(df.loc[i - 1]['经度'], df.loc[i]['经度'], df.loc[i - 1]['纬度'], df.loc[i]['纬度'])]