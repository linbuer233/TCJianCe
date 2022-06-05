# 提前选取CMA台风数据路径7-8月份，方便后续使用
import pandas as pd 
import glob

file_path=glob.glob('data\\CMABSTdata\\中国台风网台风最佳路径数据\\*')
# print(file_path[0][30:36])
for f_i in file_path:
    df=pd.read_csv(f_i,header=None,sep="\s+")
    monthlist=[]
    # 处理原始数据，添加月份列，台风名字列，方便后续的识别
    for i in range(len(df)):
        # 最后一行的处理
        if i==len(df)-1:
            df[7][i]=name
            monthlist.append(pd.to_datetime(df[0][i],format='%Y%m%d%H').month)
            break
        # 让每个台风所属的路径带有台风名字，方便后续选取7-8月份的路径数据，辨识台风
        if df[0][i]==66666:
            monthlist.append(pd.to_datetime(df[0][i+1],format='%Y%m%d%H').month)
            name=df[7][i]
            continue
        monthlist.append(pd.to_datetime(df[0][i],format='%Y%m%d%H').month)
        df[7][i]=name
    df['month']=monthlist
    # 更改名字
    df=df.rename({0:'时间',1:'强度',2:'纬度',3:'经度',4:'最低气压',5:'最大风速',6:'7',7: '名字','month':'月份'},axis=1)
    df=df.drop('7',axis=1)
    # 经度和纬度处理
    df['纬度']=df['纬度']/10
    df['经度']=df['经度']/10
    # 输出为csv文件
    df.loc[df['月份'].isin([7,8])].to_csv('data\\CMATC7-8\\'+f_i[30:36]+'.csv')