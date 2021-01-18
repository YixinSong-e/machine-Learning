import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate
def sigmoid(inx):
    if(inx >= 0):
        return 1.0/(1+math.exp(-inx))
    else:
        return math.exp(inx)/(1+math.exp(inx))

def Kernel(x,y,sigma):
    k = []
    sigma2 = 2.0 * sigma * sigma
    for i in x:
        t = np.exp(-np.sum((y - np.tile(i,(y.shape[0],1))) ** 2,axis = 1 ) / sigma2)
        k.append(t)
    return np.array(k)
def ComputeOutput(xi,weights,bias,sigma,x_train,num,kernels):
    sum = 0.0
    #for i in range(num):
        #sum += weights[i] * kernels[xi][i]
    sum += weights.dot(kernels[xi])
    sum += bias
    return sigmoid(sum)
def train(x_train,y_train,epoch,sigma,x_val,y_val):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    bias = 0  # 偏置值初始化
    alpha = np.zeros(dim)
    weights = np.zeros(num)  # 权重初始化
    learning_rate = 0.003 # 初始学习率
    kernelss = Kernel(x_val,x_train,sigma)
    kernels = Kernel(x_train,x_train,sigma)
    loss = []
    acc1 = []
    for i in range(epoch):
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        randIndex = np.random.randint(3000,size=3000)
        for j in randIndex:
            p = ComputeOutput(j,weights,bias,sigma,x_train,num,kernels)
            y = y_train[j]
            #for k in range(num):
               # weights[k] += learning_rate * (y-p) *kernels[j][k]
            weights += learning_rate * (y-p) *kernels[j]
            bias += learning_rate * (y - p) * 1
        #acc = validate(x_val,y_val,x_train,sigma,weights,bias,kernelss,500)
        if(i%10==0):
            acc = validate(x_train,y_train,x_train,sigma,weights,bias,kernels,3500)
            acc1.append(acc)
    t1 = np.linspace(5,1700,170)
    rx1 = np.array(acc1)
    f1 = interpolate.interp1d(t1,rx1,kind = 'linear')
    #f2 = interpolate.interp1d(t1,rx2,kind='cubic')
    xnew = np.linspace(5,epoch,1700)
    plt.plot(xnew,f1(xnew),'k',color='deepskyblue',label = '训练集')
    #plt.plot(xnew,f2(xnew),'k',color='deepskyblue',label = '测试集')
    plt.title('准确率随学习次数变化图')
    plt.show()

        #ls = calloss(x_train,y_train,weights,bias,kernels,sigma)
        #loss.append(ls)
    '''
    errnum = 0
    for j in range(num):
        p = ComputeOutput(j,weights,bias,sigma,x_train,num,kernels)
        y = y_train[j]
        if(p > 0.5 and y == 0):
            errnum += 1
        elif(p < 0.5 and y == 1):
            errnum += 1
    print('errnum:',errnum)
    '''
    return alpha,bias,acc1
def calloss(x_train,y_train,weights,bias,kernels,sigma):
    num = 3500
    loss = 0
    for j in range(num):
        y_pre = ComputeOutput(j,weights,bias,sigma,x_train,num,kernels)
        sig = sigmoid(y_pre)
        #print(sig)
        if(sig != 1.0 and sig != 0.0):
            loss += (-1) * (y_train[j] * np.log(sig) + (1 - y_train[j]) * np.log(1 - sig))
    return loss/num
'''
def validate(x_val, y_val,x_train,sigma, weights, bias):
    num = 500
    acc = 0
    for j in range(num):
        p = weights.dot(x_val[j,:]) +bias
        sig = sigmoid(p)
        if(sig>0.5 and y_val[j] == 1):
            acc += 1
        elif(sig <=0.5 and y_val[j] ==0):
            acc += 1
        print(sig,y_val[j])
    return acc / num
'''

# 验证模型效果
def validate(x_val, y_val,x_train,sigma, weights, bias,kernelss,num):
    #num = 500
    # loss = 0
    acc = 0
    #kernelss = Kernel(x_val,x_train,sigma)
    result = np.zeros(num)
    for j in range(num):
        p = ComputeOutput(j,weights,bias,sigma,x_train,3500,kernelss)
       # print(p,":",y_val[j])
        if p >= 0.5:
            result[j] = 1
        else:
            result[j] = 0

        if result[j] == y_val[j]:
            acc += 1.0
        # loss += (-1) * (y_val[j] * np.log(sig) + (1 - y_val[j]) * np.log(1 - sig))
    return acc / num

def analysis(loss,epoch):
    t1 = np.arange(epoch)
    t1 = t1 + 1#横坐标
    rx1 = np.array(loss)
    f1 = interpolate.interp1d(t1,rx1,kind = 'linear')
    xnew = np.linspace(1,epoch,1000)
    plt.plot(xnew,f1(xnew),'k')
    plt.title('准确率随学习次数变化图')
    plt.rcParams['font.sans-serif'] = ['SimHei'] #使中文能正常显示
    plt.rcParams['axes.unicode_minus'] = False #使符号能正常显示
    plt.rcParams['font.size'] = 16 #改变字体大小
    plt.show()
def main():
    #read data
    df = pd.read_csv('F:/syx/机器学习/结课/database.csv',header = None)
    #df = pd.read_csv('database.csv')
    df = df.fillna(0)
    ar = np.array(df)
    #deal with data
    x = ar[ : , 1:-1]
    x[:, -1] /= np.mean(x[:, -1])
    x[:, -2] /= np.mean(x[:, -2])
    y = ar[:,-1]
    #划分训练集测试集
    x_train,x_val = x[500:4000,:],x[0:500,:]
    y_train,y_val = y[500:4000],y[0:500]
    loss = []
    epoch = 1700 #训练轮
    sigma = 0.2
    plt.rcParams['font.sans-serif'] = ['SimHei'] #使中文能正常显示
    plt.rcParams['axes.unicode_minus'] = False #使符号能正常显示
    plt.rcParams['font.size'] = 16 #改变字体大小
    w,b,loss = train(x_train,y_train,epoch,sigma,x_val,y_val)
    #acc = validate(x_val,y_val,x_train,sigma,w,b)
    #print(acc)
    #analysis(loss,epoch)
    
if __name__ == '__main__':
    main()
