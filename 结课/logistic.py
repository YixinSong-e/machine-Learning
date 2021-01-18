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
def trainNormal(x_train,y_train,epoch,x_val,y_val):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    bias = 0  # 偏置值初始化
    weights = np.ones(dim)  # 权重初始化
    learning_rate = 0.1 # 初始学习率
    #learning_rate = 0.1
    reg_rate = 0.001  # 正则项系数
    loss = []#训练
    loss2 = []#测试
    acc = []#训练
    acc2 = []#测试
    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(dim)
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        for j in range(num):
            y_pre = weights.dot(x_train[j, :]) + bias
            #sig = 1 / (1 + np.exp(-y_pre))
            #sig = .5 * (1 + np.tanh(.5 * (-y_pre)))
            sig = sigmoid(y_pre)
            b_g += (-1) * (y_train[j] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j, k] + 2 * reg_rate * weights[k]
        b_g /= num
        w_g /= num

        
        #bg2_sum += b_g ** 2
        #wg2_sum += w_g ** 2
        #bias -= learning_rate / bg2_sum ** 0.5 * b_g
        #weights -= learning_rate / wg2_sum ** 0.5 * w_g
        #lrr = learning_rate / bg2_sum ** 0.5
        #learning_rates.append(lrr)
        bias -= learning_rate * b_g
        weights -= learning_rate * w_g 
        
        ls = calloss(x_train,y_train,weights,bias,3500)
        ls2 = calloss(x_val,y_val,weights,bias,500)
        loss2.append(ls2)
        loss.append(ls)
        

    return weights,bias,loss,loss2
def train(x_train,y_train,epoch,x_val,y_val):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    bias = 0  # 偏置值初始化
    weights = np.ones(dim)  # 权重初始化
    learning_rate = 0.1 # 初始学习率
    #learning_rate = 0.1
    reg_rate = 0.001  # 正则项系数
    bg2_sum = 0  # 用于存放偏置值的梯度平方和
    wg2_sum = np.zeros(dim)  # 用于存放权重的梯度平方和
    loss = []#训练
    loss2 = []#测试
    acc = []#训练
    acc2 = []#测试
    learning_rates = [1]
    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(dim)
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        for j in range(num):
            y_pre = weights.dot(x_train[j, :]) + bias
            #sig = 1 / (1 + np.exp(-y_pre))
            #sig = .5 * (1 + np.tanh(.5 * (-y_pre)))
            sig = sigmoid(y_pre)
            b_g += (-1) * (y_train[j] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j, k] + 2 * reg_rate * weights[k]
        b_g /= num
        w_g /= num

        
        bg2_sum += b_g ** 2
        wg2_sum += w_g ** 2
        bias -= learning_rate / bg2_sum ** 0.5 * b_g
        weights -= learning_rate / wg2_sum ** 0.5 * w_g
        lrr = learning_rate / bg2_sum ** 0.5
        learning_rates.append(lrr)
        #bias -= learning_rate * b_g
        #weights -= learning_rate * w_g
        if(i%5==0):
            acc1 = validate(x_val,y_val,weights,bias,500)
            acc2.append(acc1)
            acc1 = validate(x_train,y_train,weights,bias,3500)
            acc.append(acc1)
        
        
        ls = calloss(x_train,y_train,weights,bias,3500)
        ls2 = calloss(x_val,y_val,weights,bias,500)
        loss2.append(ls2)
        loss.append(ls)
        
    #print(acc2)
        
    '''
    t1 = np.linspace(5,100,20)
    rx1 = np.array(acc)
    rx2 = np.array(acc2)
    f1 = interpolate.interp1d(t1,rx1,kind = 'cubic')
    f2 = interpolate.interp1d(t1,rx2,kind='cubic')
    xnew = np.linspace(5,epoch,1000)
    plt.plot(xnew,f1(xnew),'k--',color='coral',label = '训练集')
    plt.plot(xnew,f2(xnew),'k',color='deepskyblue',label = '测试集')
    plt.title('准确率随学习次数变化图')
    '''
    return weights,bias,loss,loss2
def trainRMS(x_train,y_train,epoch,x_val,y_val):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    bias = 0  # 偏置值初始化
    weights = np.ones(dim)  # 权重初始化
    learning_rate = 0.1 # 初始学习率
    #learning_rate = 0.1
    reg_rate = 0.001  # 正则项系数
    bg2_sum = 1  # 用于存放偏置值的梯度平方和
    wg2_sum = np.ones(dim)  # 用于存放权重的梯度平方和
    loss = []#训练
    loss2 = []#测试
    acc = []#训练
    acc2 = []#测试
    beta2 = 0.99
    #learning_rates = [1]
    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(dim)
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        for j in range(num):
            y_pre = weights.dot(x_train[j, :]) + bias
            #sig = 1 / (1 + np.exp(-y_pre))
            #sig = .5 * (1 + np.tanh(.5 * (-y_pre)))
            sig = sigmoid(y_pre)
            b_g += (-1) * (y_train[j] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j, k] + 2 * reg_rate * weights[k]
        b_g /= num
        w_g /= num

        
        bg2_sum = beta2 * bg2_sum + (1-beta2) * b_g ** 2
        wg2_sum = beta2 * wg2_sum + (1-beta2) * w_g ** 2
        bias -= learning_rate / bg2_sum ** 0.5 * b_g
        weights -= learning_rate / wg2_sum ** 0.5 * w_g
        #lrr = learning_rate / bg2_sum ** 0.5
        #learning_rates.append(lrr)
        #bias -= learning_rate * b_g
        #weights -= learning_rate * w_g
        if(i%5==0):
            acc1 = validate(x_val,y_val,weights,bias,500)
            acc2.append(acc1)
            acc1 = validate(x_train,y_train,weights,bias,3500)
            acc.append(acc1)
        
        
        ls2 = calloss(x_val,y_val,weights,bias,500)
        loss2.append(ls2)
        
    #print(acc2)
        
    return weights,bias,loss2
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
    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(dim)
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        randIndex = np.random.randint(3500,size=400)
        for j in range(batch_size):
            y_pre = weights.dot(x_train[randIndex[j], :]) + bias
            #sig = 1 / (1 + np.exp(-y_pre))
            #sig = .5 * (1 + np.tanh(.5 * (-y_pre)))
            sig = sigmoid(y_pre)
            b_g += (-1) * (y_train[randIndex[j]] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[randIndex[j]] - sig) * x_train[randIndex[j], k] + 2 * reg_rate * weights[k]
        b_g /= batch_size
        w_g /= batch_size

        bias -= learning_rate * b_g
        weights -= learning_rate * w_g
        ls = calloss(x_val,y_val,weights,bias,500)
        loss.append(ls)

    return weights,bias,loss
def trainSGDM(x_train,y_train,epoch,learning_rate,x_val,y_val):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    beta1 = 0.9
    bias = 0  # 偏置值初始化
    weights = np.ones(dim)  # 权重初始化
    #learning_rate = 1 # 初始学习率
    #learning_rate = 1
    reg_rate = 0.001  # 正则项系数
    batch_size = 400
    loss = []
    mb_g = 0
    mw_g = 0
    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(dim)
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        randIndex = np.random.randint(3500,size=400)
        for j in range(batch_size):
            y_pre = weights.dot(x_train[randIndex[j], :]) + bias
            #sig = 1 / (1 + np.exp(-y_pre))
            #sig = .5 * (1 + np.tanh(.5 * (-y_pre)))
            sig = sigmoid(y_pre)
            b_g += (-1) * (y_train[randIndex[j]] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[randIndex[j]] - sig) * x_train[randIndex[j], k] + 2 * reg_rate * weights[k]
        b_g /= batch_size
        w_g /= batch_size
        mb_g = beta1 * mb_g + (1-beta1) * b_g
        mw_g = beta1 * mw_g + (1-beta1) * w_g
        bias -= learning_rate * mb_g
        weights -= learning_rate * mw_g
        ls = calloss(x_val,y_val,weights,bias,500)
        loss.append(ls)

    return weights,bias,loss
def trainNAG(x_train,y_train,epoch,learning_rate,x_val,y_val):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    beta1 = 0.9
    bias = 0  # 偏置值初始化
    weights = np.ones(dim)  # 权重初始化
    #learning_rate = 1 # 初始学习率
    #learning_rate = 1
    reg_rate = 0.001  # 正则项系数
    bg2_sum = 0  # 用于存放偏置值的梯度平方和
    wg2_sum = np.zeros(dim)  # 用于存放权重的梯度平方和
    batch_size = 400
    loss = []
    mb_g = 0
    mw_g = 0
    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(dim)
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        randIndex = np.random.randint(3500,size=400)
        newweights = weights - learning_rate * mw_g 
        newbias = bias - learning_rate * mb_g
        for j in range(batch_size):
            y_pre = newweights.dot(x_train[randIndex[j], :]) + newbias
            #sig = 1 / (1 + np.exp(-y_pre))
            #sig = .5 * (1 + np.tanh(.5 * (-y_pre)))
            sig = sigmoid(y_pre)
            b_g += (-1) * (y_train[randIndex[j]] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[randIndex[j]] - sig) * x_train[randIndex[j], k] + 2 * reg_rate * weights[k]
        b_g /= batch_size
        w_g /= batch_size
        mb_g = beta1 * mb_g + (1-beta1) * b_g
        mw_g = beta1 * mw_g + (1-beta1) * w_g
        bias -= learning_rate * mb_g
        weights -= learning_rate * mw_g
        ls = calloss(x_val,y_val,weights,bias,500)
        loss.append(ls)

    return weights,bias,loss
def trainAdam(x_train,y_train,epoch,x_val,y_val):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    bias = 0  # 偏置值初始化
    weights = np.ones(dim)  # 权重初始化
    learning_rate = 0.1 # 初始学习率
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
        b_g = 0
        w_g = np.zeros(dim)
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        for j in range(num):
            y_pre = weights.dot(x_train[j, :]) + bias
            #sig = 1 / (1 + np.exp(-y_pre))
            #sig = .5 * (1 + np.tanh(.5 * (-y_pre)))
            sig = sigmoid(y_pre)
            b_g += (-1) * (y_train[j] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j, k] + 2 * reg_rate * weights[k]
        b_g /= num
        w_g /= num

        
        mb_t = beta1 * mb_t + (1-beta1) * b_g
        mw_t = beta1 * mw_t + (1-beta1) * w_g
        vb_t = beta2 * vb_t + (1-beta2) * b_g ** 2
        vw_t = beta2 * vw_t + (1-beta2) * w_g ** 2

        bias -= learning_rate / vb_t ** 0.5 * mb_t
        weights -= learning_rate / vw_t ** 0.5 * mw_t
        #lrr = learning_rate / bg2_sum ** 0.5
        #learning_rates.append(lrr)
        #bias -= learning_rate * b_g
        #weights -= learning_rate * w_g
        if(i%5==0):
            acc1 = validate(x_val,y_val,weights,bias,500)
            acc2.append(acc1)
            acc1 = validate(x_train,y_train,weights,bias,3500)
            acc.append(acc1)
        
        
        ls = calloss(x_train,y_train,weights,bias,3500)
        ls2 = calloss(x_val,y_val,weights,bias,500)
        loss2.append(ls2)
        loss.append(ls)
        
    print(acc2)
        
    '''
    t1 = np.linspace(5,100,20)
    rx1 = np.array(acc)
    rx2 = np.array(acc2)
    f1 = interpolate.interp1d(t1,rx1,kind = 'cubic')
    f2 = interpolate.interp1d(t1,rx2,kind='cubic')
    xnew = np.linspace(5,epoch,1000)
    plt.plot(xnew,f1(xnew),'k--',color='coral',label = '训练集')
    plt.plot(xnew,f2(xnew),'k',color='deepskyblue',label = '测试集')
    plt.title('准确率随学习次数变化图')
    '''
    return weights,bias,loss,loss2
def trainNAdam(x_train,y_train,epoch,x_val,y_val):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    bias = 0  # 偏置值初始化
    weights = np.ones(dim)  # 权重初始化
    learning_rate = 0.1 # 初始学习率
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
    vb_t = 0.001
    vw_t = np.ones(dim)
    vw_t = vw_t * 0.001
    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(dim)
        newweights = weights - learning_rate * mw_t / vw_t ** 0.5
        newbias = bias - learning_rate * mb_t / vb_t ** 0.5
        # 在所有数据上计算梯度，梯度计算时针对损失函数求导
        for j in range(num):
            y_pre = newweights.dot(x_train[j, :]) + newbias
            #sig = 1 / (1 + np.exp(-y_pre))
            #sig = .5 * (1 + np.tanh(.5 * (-y_pre)))
            sig = sigmoid(y_pre)
            b_g += (-1) * (y_train[j] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j, k] + 2 * reg_rate * newweights[k]
        b_g /= num
        w_g /= num

        
        mb_t = beta1 * mb_t + (1-beta1) * b_g
        mw_t = beta1 * mw_t + (1-beta1) * w_g
        vb_t = beta2 * vb_t + (1-beta2) * b_g ** 2
        vw_t = beta2 * vw_t + (1-beta2) * w_g ** 2

        bias -= learning_rate / vb_t ** 0.5 * mb_t
        weights -= learning_rate / vw_t ** 0.5 * mw_t
        #lrr = learning_rate / bg2_sum ** 0.5
        #learning_rates.append(lrr)
        #bias -= learning_rate * b_g
        #weights -= learning_rate * w_g
        if(i%5==0):
            acc1 = validate(x_val,y_val,weights,bias,500)
            acc2.append(acc1)
            acc1 = validate(x_train,y_train,weights,bias,3500)
            acc.append(acc1)
        
        
        ls = calloss(x_train,y_train,weights,bias,3500)
        ls2 = calloss(x_val,y_val,weights,bias,500)
        loss2.append(ls2)
        loss.append(ls)
        
    print(acc2)
        
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
        sig = sigmoid(y_pre)
        if sig >= 0.5:
            result[j] = 1
        else:
            result[j] = 0

        if result[j] == y_val[j]:
            acc += 1.0
        # loss += (-1) * (y_val[j] * np.log(sig) + (1 - y_val[j]) * np.log(1 - sig))
    return acc / num
def analysis(loss,epoch,flags):
    t1 = np.arange(epoch)
    t1 = t1 + 1#横坐标
    rx1 = np.array(loss)
    f1 = interpolate.interp1d(t1,rx1,kind = 'linear')
    xnew = np.linspace(1,epoch,1000)
    if(flags == 0):
        plt.plot(xnew,f1(xnew),'k',label = '训练')
    else:
        plt.plot(xnew,f1(xnew),'k--',color='deepskyblue',label = '测试')
    plt.title('损失函数随学习次数变化图')
    plt.rcParams['font.sans-serif'] = ['SimHei'] #使中文能正常显示
    plt.rcParams['axes.unicode_minus'] = False #使符号能正常显示
    plt.rcParams['font.size'] = 16 #改变字体大小
    
    #plt.show()
def analysisAcc(acc,epoch,flags):
    t1 = np.arange(epoch)
    t1 = t1 + 1#横坐标
    rx1 = np.array(acc)
    f1 = interpolate.interp1d(t1,rx1,kind = 'linear')
    xnew = np.linspace(1,epoch,1000)
    if(flags == 0):
        plt.plot(xnew,f1(xnew),'k',label = 'training set')
    else:
        plt.plot(xnew,f1(xnew),'k--',label = 'test set')
    plt.title('准确率随学习次数变化图')
def analysisLoss(loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,epoch):
    t1 = np.arange(epoch)
    t1 = t1 + 1#横坐标
    rx1 = np.array(loss1)
    rx2 = np.array(loss2)
    rx3 = np.array(loss3)
    rx4 = np.array(loss4)
    rx5 = np.array(loss5)
    rx6 = np.array(loss6)
    rx7 = np.array(loss7)
    rx8 = np.array(loss8)
    f1 = interpolate.interp1d(t1,rx1,kind = 'linear')
    f2 = interpolate.interp1d(t1,rx2,kind = 'linear')
    f3 = interpolate.interp1d(t1,rx3,kind = 'linear')
    f4 = interpolate.interp1d(t1,rx4,kind = 'linear')
    f5 = interpolate.interp1d(t1,rx5,kind = 'linear')
    f6 = interpolate.interp1d(t1,rx6,kind = 'linear')
    f7 = interpolate.interp1d(t1,rx7,kind = 'linear')
    f8 = interpolate.interp1d(t1,rx8,kind = 'linear')
    xnew = np.linspace(1,epoch,1000)
    plt.plot(xnew,f1(xnew),'k',color='coral',label = 'Adgrad')
    plt.plot(xnew,f2(xnew),'k--',color='deepskyblue',label = 'Adam')
    plt.plot(xnew,f3(xnew),'k--',color='tan',label = 'normal')
    plt.plot(xnew,f4(xnew),'k-',color='blueviolet',label = 'SGD')
    plt.plot(xnew,f5(xnew),'k-',color='orchid',label = 'SGDM')
    plt.plot(xnew,f6(xnew),'k-',color='gold',label = 'NAG')
    plt.plot(xnew,f7(xnew),'k',color='lime',label = 'RMS')
    plt.plot(xnew,f8(xnew),'k',color='magenta',label = 'nAdam')
    plt.title('不同优化方式下loss函数变化图')
    
def main():
    #read data
    df = pd.read_csv('F:/syx/机器学习/结课/database.csv',header = None)
    #df = pd.read_csv('database.csv')
    df = df.fillna(0)
    ar = np.array(df)
    plt.rcParams['font.sans-serif'] = ['SimHei'] #使中文能正常显示
    plt.rcParams['axes.unicode_minus'] = False #使符号能正常显示
    plt.rcParams['font.size'] = 16 #改变字体大小
    #deal with data
    x = ar[ : , 1:-1]
    x[:, -1] /= np.mean(x[:, -1])
    x[:, -2] /= np.mean(x[:, -2])
    y = ar[:,-1]
    #划分训练集测试集
    x_train,x_val = x[0:3500,:],x[3500:4000,:]
    y_train,y_val = y[0:3500],y[3500:4000]
    #loss = []
    epoch = 100 #训练轮
    w,b,lossn,loss2n = trainNormal(x_train,y_train,epoch,x_val,y_val)
    w,b,lossdm,loss2dm = trainAdam(x_train,y_train,epoch,x_val,y_val)
    w,b,lossndm,loss2ndm = trainNAdam(x_train,y_train,epoch,x_val,y_val)
    w,b,loss,loss2 = train(x_train,y_train,epoch,x_val,y_val)
    w,b,lossSgd = trainSGD(x_train,y_train,epoch,0.1,x_val,y_val)
    w,b,lossSgdM = trainSGDM(x_train,y_train,epoch,0.1,x_val,y_val)
    w,b,lossNAG = trainSGDM(x_train,y_train,epoch,0.1,x_val,y_val)
    w,b,lossRMS = trainRMS(x_train,y_train,epoch,x_val,y_val)
    #acc = validate(x_val,y_val,w,b)
    #print(acc)
    analysisLoss(loss2,loss2dm,loss2n,lossSgd,lossSgdM,lossNAG,lossRMS,loss2ndm,epoch)
    #analysis(loss,epoch,0)
    #analysis(loss2,epoch,1)
    #analysis(lossdm,epoch,0)
    #analysis(loss2dm,epoch,1)
    plt.legend(loc=0,ncol=1)
    plt.show()
    
if __name__ == '__main__':
    main()
