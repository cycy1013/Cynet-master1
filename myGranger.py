#####################################################################################
# 构造格兰杰因果关系对，x是因，y是果，y_not不是x的果
import numpy as np
np.random.seed(1)
x = np.random.standard_normal(100)
print("x:\n",x)
np.random.seed(12)
#np.r_: 就是把两矩阵上下相加，要求列数相等
#np.r_[0, x[0:-1]]:效果就是在最前面添加0，把最后一个元素去掉
y = np.r_[0, x[0:-1]]  + np.random.standard_normal(x.shape)  # x -> y，y滞后于x，所以y是x的果
#print("y:\n",y)
y_not = x + np.random.standard_normal(x.shape)  # y和x同步，所以y不是x的果


#####################################################################################
# 绘制散点图展示x、y以及x、y_not
import matplotlib.pyplot as plt
plt.plot(x, y, marker='o', ls='')
#plt.plot(x, y_not, marker='o', ls='')
#plt.show()
#####################################################################################
# 检测平稳性
from statsmodels.tsa.stattools import adfuller as ADF
pv_x = ADF(x)
pv_y = ADF(y)
pv_y_not = ADF(y_not)

print('The ADF Statistic of data: %f' % pv_x[0]) #-7.158368
print('The p value of data: %f' % pv_x[1]) #0.000000   结论:是平稳序列

print('The ADF Statistic of data: %f' % pv_y[0]) #-10.149275
print('The p value of data: %f' % pv_y[1]) #0.000000   结论:是平稳序列

print('The ADF Statistic of data: %f' % pv_y_not[0]) #-6.781555
print('The p value of data: %f' % pv_y_not[1]) #0.000000   结论:是平稳序列

#####################################################################################
# 检测格兰杰因果关系
from statsmodels.tsa.stattools import grangercausalitytests
# 若p值小于0.05，则该维度数据对预测有帮助
x_y = grangercausalitytests(np.c_[y, x], maxlag=1)  # 判断是不是 x -> y
y_x = grangercausalitytests(np.c_[x, y], maxlag=1)  # 判断是不是 y -> x
#
x_y_not_x = grangercausalitytests(np.c_[y_not, x], maxlag=1)  # 判断是不是 x -> y_not
y_not_x_x = grangercausalitytests(np.c_[x, y_not], maxlag=1)  # 判断是不是 y_not -> x
