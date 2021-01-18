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

def trainAdam(x_train,y_train,epoch,x_val,y_val):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    bias = 0  # 偏置值初始化
    weights = np.ones(dim)  # 权重初始化
    learning_rate = 0.01 # 初始学习率
    #learning_rate = 0.1
    reg_rate = 0.001  # 正则项系数
    beta1 = 0.9
    beta2 = 0.95
    loss = []#训练
    loss2 = []#测试
    acc = []#训练
    acc2 = []#测试
    mb_t = 0
    mw_t = np.zeros(dim)
    vb_t = 0
    vw_t = np.zeros(dim)
    for i in range(epoch):
        #b_g = 0
        #w_g = np.zeros(dim)
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        for j in range(num):
            b_g = 0
            w_g = np.zeros(dim)
            y_pre = weights.dot(x_train[j, :]) + bias
            #sig = 1 / (1 + np.exp(-y_pre))
            #sig = .5 * (1 + np.tanh(.5 * (-y_pre)))
            sig = sigmoid(y_pre)
            b_g += (-1) * (y_train[j] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j, k] + 2 * reg_rate * weights[k]
        #b_g /= num
        #w_g /= num

        
        #mb_t = beta1 * mb_t + (1-beta1) * b_g
        #mw_t = beta1 * mw_t + (1-beta1) * w_g
        #vb_t = beta2 * vb_t + (1-beta2) * b_g ** 2
        #vw_t = beta2 * vw_t + (1-beta2) * w_g ** 2

            bias -= learning_rate * b_g
            weights -= learning_rate * w_g
        #lrr = learning_rate / bg2_sum ** 0.5
        #learning_rates.append(lrr)
        #bias -= learning_rate * b_g
        #weights -= learning_rate * w_g
            if(j%5==0):
                acc1 = validate(x_val,y_val,weights,bias,1000)
                acc2.append(acc1)
                acc1 = validate(x_train,y_train,weights,bias,3000)
                acc.append(acc1)
        
        
            ls = calloss(x_train,y_train,weights,bias,3000)
            ls2 = calloss(x_val,y_val,weights,bias,1000)
            loss2.append(ls2)
            loss.append(ls)
        
    print(acc2)
        
    
    t1 = np.linspace(5,200,40)
    rx1 = np.array(acc)
    rx2 = np.array(acc2)
    f1 = interpolate.interp1d(t1,rx1,kind = 'cubic')
    f2 = interpolate.interp1d(t1,rx2,kind='cubic')
    xnew = np.linspace(5,epoch,1000)
    plt.plot(xnew,f1(xnew),'k--',color='coral',label = '训练集')
    plt.plot(xnew,f2(xnew),'k',color='deepskyblue',label = '测试集')
    plt.title('准确率随学习次数变化图')
    plt.xlabel('训练次数')
    plt.ylabel('准确率')
    plt.legend()
    plt.show()
    
    return weights,bias,loss,loss2
def calloss(x_train,y_train,weights,bias,num):
    loss = 0
    for j in range(num):
        y_pre = weights.dot(x_train[j,:])+bias
        sig = sigmoid(y_pre)
        #print(sig)
        if(sig != 1.0 and sig != 0.0):
            loss += (-1) * (y_train[j] * np.log(sig) + (1 - y_train[j]) * np.log(1 - sig))
    loss /= num
    loss += 0.001 * 0.5 * np.dot(weights,weights)
    loss += 0.001 * 0.5 * bias * bias
    return loss

# 验证模型效果
def validate(x_val, y_val, weights, bias,num):
    # loss = 0
    acc = 0
    result = np.zeros(num)
    for j in range(num):
        y_pre = weights.dot(x_val[j, :]) + bias
        sig = 1 / (1 + np.exp(-y_pre))
        if sig >= 0.5:
            result[j] = 1
        else:
            result[j] = 0

        if result[j] == y_val[j]:
            acc += 1.0
        # loss += (-1) * (y_val[j] * np.log(sig) + (1 - y_val[j]) * np.log(1 - sig))
    return acc / num
def analysis(loss,loss2,epoch):
    t1 = np.arange(epoch)
    t1 = t1 + 1#横坐标
    rx1 = np.array(loss)
    rx2 = np.array(loss2)
    f1 = interpolate.interp1d(t1,rx1,kind = 'linear')
    f2 = interpolate.interp1d(t1,rx2,kind = 'linear')
    xnew = np.linspace(1,epoch,1000)
    plt.plot(xnew,f1(xnew),'k',color='coral',label='训练集')
    plt.plot(xnew,f2(xnew),'k--',color='deepskyblue',label='测试集')
    
    plt.title('损失函数随学习次数变化图')
    plt.legend()
    plt.show()
def trainSGD(x_train,y_train,epoch,learning_rate,x_val,y_val):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    bias = 0  # 偏置值初始化
    weights = np.ones(dim)  # 权重初始化
    #learning_rate = 1 # 初始学习率
    #learning_rate = 1
    reg_rate = 0.001  # 正则项系数
    bg2_sum = 0  # 用于存放偏置值的梯度平方和
    wg2_sum = np.zeros(dim)  # 用于存放权重的梯度平方和
    batch_size = 400
    loss = []
    loss2 = []
    acc = []
    acc2 = []
    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(dim)
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        #randIndex = np.random.randint(3000,size=400)
        for j in range(num):
            y_pre = weights.dot(x_train[j, :]) + bias
            #sig = 1 / (1 + np.exp(-y_pre))
            #sig = .5 * (1 + np.tanh(.5 * (-y_pre)))
            sig = sigmoid(y_pre)
            b_g += (-1) * (y_train[j] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j, k] + 2 * reg_rate * weights[k]
        b_g /= batch_size
        w_g /= batch_size

        bias -= learning_rate * b_g
        weights -= learning_rate * w_g
        if(i%5==0):
            acc1 = validate(x_val,y_val,weights,bias,1000)
            acc2.append(acc1)
            acc1 = validate(x_train,y_train,weights,bias,3000)
            acc.append(acc1)
        ls = calloss(x_val,y_val,weights,bias,1000)
        ls2 = calloss(x_train,y_train,weights,bias,3000)
        loss.append(ls)
        loss2.append(ls2)
    '''
    t1 = np.linspace(5,1000,200)
    rx1 = np.array(acc)
    rx2 = np.array(acc2)
    f1 = interpolate.interp1d(t1,rx1,kind = 'linear')
    f2 = interpolate.interp1d(t1,rx2,kind='linear')
    xnew = np.linspace(5,epoch,1000)
    plt.plot(xnew,f1(xnew),'k--',color='coral',label = '训练集')
    plt.plot(xnew,f2(xnew),'k',color='deepskyblue',label = '测试集')
    plt.title('准确率随学习次数变化图')
    plt.xlabel('训练次数')
    plt.ylabel('准确率')
    plt.legend()
    plt.show()
    '''
    return weights,bias,loss2,loss
def main():
    #read data
    df = pd.read_csv('F:/syx/机器学习/结课/database.csv',header = None)
    plt.rcParams['font.sans-serif'] = ['SimHei'] #使中文能正常显示
    plt.rcParams['axes.unicode_minus'] = False #使符号能正常显示
    plt.rcParams['font.size'] = 16 #改变字体大小
    #df = pd.read_csv('database.csv')
    df = df.fillna(0)
    ar = np.array(df)
    #deal with data
    x = ar[ : , 1:-1]
    x[:, -1] /= np.mean(x[:, -1])
    x[:, -2] /= np.mean(x[:, -2])
    y = ar[:,-1]
    #划分训练集测试集
    x_train,x_val = x[1000:4000,:],x[:1000,:]
    y_train,y_val = y[1000:4000],y[:1000]
    loss = []
    loss2 = []
    #acc = []
    epoch = 4000 #训练轮
    learning_rate = 0.006
    #w,b,loss,loss2 = trainAdam(x_train,y_train,epoch,x_val,y_val)
    w,b,loss,loss2 = trainSGD(x_train,y_train,epoch,learning_rate,x_val,y_val)
    analysis(loss,loss2,epoch)
    acc0 = validate(x_train,y_train,w,b,3000)
    print(acc0)
    acc1 = validate(x_val,y_val,w,b,1000)
    print(acc1)
    '''
    for i in range(2):
        w,b = train(x_train,y_train,epoch,learning_rate)
        acc1 = validate(x_val,y_val,w,b)
        acc.append(acc1)
        learning_rate+=1
    print(acc)
    '''
    #t1 = np.linspace(0.001,2.001,2)
    #rx1 = np.array(acc)
    #plt.plot(t1,rx1,kind = 'k')
    #plt.title('不同学习率梯度下降收敛后的准确率')
    #plt.rcParams['font.sans-serif'] = ['SimHei'] #使中文能正常显示
    #plt.rcParams['axes.unicode_minus'] = False #使符号能正常显示
    #plt.rcParams['font.size'] = 16 #改变字体大小
    #plt.show()
    #w,b,loss = train(x_train,y_train,epoch)
    #acc = validate(x_val,y_val,w,b)
    #print(acc)
    #analysis(loss,epoch)
    
if __name__ == '__main__':
    main()
