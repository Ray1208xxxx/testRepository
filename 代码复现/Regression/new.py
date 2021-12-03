import numpy as np
import pandas as pd
import scipy.stats as stats
from math import *

#x取1~100,y取某一个非线性函数
x = np.arange(1,101)
x=np.array([float(i) for i in x])
y =x+[10*sin(0.3*i)for i in x]+stats.norm.rvs(size=100, loc=0, scale=1.5)

#先加载画图需要用的包
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
import seaborn as sns

plt.figure(figsize=(12,6))
plt.scatter(x,y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
plt.figure(figsize=(12,6))
yHatLinear=intercept+slope*x
plt.plot(x,yHatLinear,'r')
plt.scatter(x,y)
print('y='+str(intercept)+'+'+str(slope)+'x')

#局部加权线性回归里的权重的平方根
def get_sqrtW(x0,k):    #x0处其它样本点的权重
    w=np.zeros(len(x))
    for i in range(len(x)):
        w[i]=exp(-(x[i]-x0)**2/(2*k*k))
    w=np.array([sqrt(i) for i in w])
    return w
import statsmodels.api as sm
#把整个过程定义成一个函数，方便改参数进行研究
def get_yHat2(k):
    yHat2=np.zeros(len(x))
    for i in range(len(x)):
        #把加权最小二乘转化为普通最小二乘，详情请看第二部分
        w=get_sqrtW(x[i],k)
        x2=w*x
        x2=x2[x2>0]      #去掉样本权重为0的样本
        y2=w*y
        y2=y2[y2>0]
        X=np.zeros((1,len(x2)))
        X[0]=x2
        X=X.T
        X = sm.add_constant(X,has_constant='skip')
        X[:,0]=w[w>0]
        Y=y2
        model = sm.OLS(Y, X)
        results = model.fit()
        a = results.params[0]
        b = results.params[1]
        yHat2[i]=a+b*x[i]      #得到xi处的估计值
    return yHat2
yHat2=get_yHat2(100000)  #ｋ取100000
plt.figure(figsize=(12,6))
plt.plot(x,yHat2,'r')
plt.scatter(x,y)
data=pd.DataFrame()
data['y']=y
data['yHatLinear']=yHatLinear
data['yHat2']=yHat2
data.head()
yHat2=get_yHat2(10)
plt.figure(figsize=(12,6))
plt.plot(x,yHat2,'r')
plt.scatter(x,y)
plt.title('k=10')

yHat2=get_yHat2(1)
plt.figure(figsize=(12,6))
plt.plot(x,yHat2,'r')
plt.scatter(x,y)
plt.title('k=1')

yHat2=get_yHat2(0.1)
plt.figure(figsize=(12,6))
plt.plot(x,yHat2,'r')
plt.scatter(x,y)
plt.title('k=0.1')