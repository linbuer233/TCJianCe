# 输入数据的形状（N，Cin，Hin，Win），输出的形状（N，Cout，Hout，Wout） 依次为，批次||通道||高||宽
Hin = 81
Win = 81
pad = [0,0]  # 补0操作，默认为0
dilation = [1,1]  # 卷积核中每个元素的 间隔， 默认为 1
kernel_size = [3, 3]  # 卷积核大小，h*w
stride = [1,1]  # 卷积核步长 默认为1

Hout=(Hin+2*pad[0]-dilation[0]*(kernel_size[0]-1)-1)/(stride[0])+1
Wout=(Win+2*pad[1]-dilation[1]*(kernel_size[1]-1)-1)/(stride[1])+1

print(Hout,Wout)