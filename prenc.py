# 合并数据，方便后续调用
import xarray as xr 

for i in ["RV","T","U","V"]:
    var=i
    name1="data\\"+var+"_2017-2018_7-8.nc"
    name2="data\\"+var+"_2019-2020_7-8.nc"
    name3="data\\"+var+"_2017-2020_7-8.nc"
    ds1=xr.open_dataset(name1)
    ds2=xr.open_dataset(name2)
    xr.merge([ds1,ds2]).to_netcdf(name3)