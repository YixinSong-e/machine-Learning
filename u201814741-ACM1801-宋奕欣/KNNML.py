import os
import numpy as np
from matplotlib import pyplot as plt
import operator
IMAGE_ROW = 28
IMAGE_COL = 28
IMAGE_SIZE = 28*28

def readFileFirst(filename):
    dimension = []
    with open(filename,'rb') as pf:
        data = pf.read(4)
        magic_num = int.from_bytes(data,byteorder='big')

        dimension_cnt = magic_num & 0xff

        for i in range(dimension_cnt):
            data = pf.read(4)
            dms = int.from_bytes(data,byteorder='big')
            dimension.append(dms)
    return dimension
def get_head_length(dimension):
    return 4*len(dimension)+4

def read_image_vector(filename,head_len,offset,amount):
    image_mat=np.zeros((amount,IMAGE_SIZE),dtype=np.uint8)
    with open(filename,'rb') as pf:
        #magic_num的长度为4，dimension_cnt单个长度为4,前面的number个长度为28*28*offset   
        pf.seek(head_len+IMAGE_SIZE*offset) 
        
        for ind in range(amount):
            image = np.zeros((1,IMAGE_SIZE),dtype=np.uint8)
            for row in range(IMAGE_SIZE):
                data = pf.read(1)
                pix = int.from_bytes(data,byteorder='big')，
                
                if pix >127:image[0][row] = 1
            image_mat[ind,:]=image
            #print('read_image_vector：当前进度%0.2f%%'%(ind*100.0/amount),end='\r')
        print('图像解析完毕')
    return image_mat
def read_label_vector(filename,head_len,offset,amount):
    label_list=[]
    
    with open(filename,'rb') as pf:
        #pf 指向label的第number个数据,magic_num的长度为4，dimension_cnt单个长度为4
        pf.seek(head_len+offset) 
        
        for ind in range(amount):
            data = pf.read(1)
            label = int.from_bytes(data,byteorder='big') 
            label_list.append(label)
            
        print('标签解析完毕')
    
    return label_list
def read_image_label_vector(image_file,label_file,offset,amount):
    
    image_dim = readFileFirst(image_file)
    label_dim = readFileFirst(label_file)
    
    #判断样本中的image和label是否一致
    image_amount = image_dim[0]
    label_amount = label_dim[0]

    
    #获取样本image和label的头文件长度
    image_head_len = get_head_length(image_dim)
    label_head_len = get_head_length(label_dim)
    
    #得到image和label的向量
    image_mat = read_image_vector(image_file,image_head_len,offset,amount)
    label_list = read_label_vector(label_file,label_head_len,offset,amount)
    
    return image_mat,label_list
def knn_classify(test_data, train_dataset, train_label, k):
    train_dataset_amount = train_dataset.shape[0]#行数，也即训练样本的的个数，shape[1]为列数
    #将输入test_data变成了和train_dataset行列数一样的矩阵
    test_rep_mat =  np.tile(test_data, (train_dataset_amount,1))#tile(mat,(x,y)) Array类 mat 沿着行重复x次，列重复y次
    diff_mat = test_rep_mat - train_dataset

    sq_diff_mat = diff_mat**2  

    sq_dist = sq_diff_mat.sum(axis=1)

    dist_index = sq_dist.argsort()   
    
    for i in range(5):
        label = train_label[dist_index[i]]
        show_picture(train_dataset,label,dist_index[i])
        os.system("pause")
    
    class_count={}        
    for i in range(k):
        label = train_label[dist_index[i]]
        #如果属于某个类，在该类的基础上加1，相当于增加其权重，如果不是某个类则新建字典的一个key并且等于1
        class_count[label] = class_count.get(label,0) + 1
    #降序排列
    class_count_list = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    #print('排序后的分类结果：',class_count_list)
    return class_count_list[0][0]

def show_picture(image_mat,label,number):
    image = np.zeros((IMAGE_ROW,IMAGE_COL),dtype = np.uint8)
    for i in range(IMAGE_ROW):
        for j in range(IMAGE_COL):
            image[i][j] = image_mat[number][i*IMAGE_ROW+j]
    plt.imshow(image)
    plt.title('picture number = %d,label = %d' %(number,label))
    plt.show()
def run_KNN(train_image_mat,train_label_list,test_image_mat,test_label_list,k):
    err_count = 0.0
    for i in range(len(test_image_mat)):
                #print('当前进度：%2.2f%%'%(100.0*i/len(test_image_mat)))
                #利用knn算法进行分类
                show_picture(test_image_mat,test_label_list[i],i)
                class_result = knn_classify(test_image_mat[i], train_image_mat, train_label_list, k)#计算分类结果
                if (class_result != test_label_list[i]):
                    err_count += 1.0
    print( "\n总错误数: %d" % err_count)
    error_rate = 100.0*err_count/len(test_image_mat)
    print( "总错误率: %2.2f%%" % (error_rate))
    return error_rate

if __name__ == '__main__':
    train_image_file = 'F:\\syx\\机器学习\\KNN\\train-images.idx3-ubyte'
    train_label_file = 'F:\\syx\\机器学习\\KNN\\train-labels.idx1-ubyte'
    test_image_file = 'F:\\syx\\机器学习\\KNN\\t10k-images.idx3-ubyte'
    test_label_file = 'F:\\syx\\机器学习\\KNN\\t10k-labels.idx1-ubyte'
    train_image_mat, train_label_list  = read_image_label_vector(train_image_file,train_label_file,0,50000)
    traineva_image_mat, traineva_label_list = read_image_label_vector(train_image_file,train_label_file,50000,10000)
    err_count = 0.0
    '''
    for i in range(len(train_image_mat)):
                print('当前进度：%2.2f%%'%(100.0*i/len(train_image_mat)))
                #利用knn算法进行分类
                class_result = knn_classify(train_image_mat[i], train_image_mat, train_label_list, 5)#计算分类结果
                print( "第 %d 张图片, 分类器结果: %d, 实际值: %d" % (i,class_result, traineva_label_list[i]),end=' ')
                #打印错误率
                print('当前错误率：%2.2f%%' % (100.0*err_count/(i+0.01)))
    print( "\n总错误数: %d" % err_count)
    print( "总错误率: %2.2f%%" % (100.0*err_count/len(train_image_mat)))
    '''
    amount = 10000
    K = [1,5,10,20,50,100,120]
    #error_rate = []
    test_image_mat, test_label_list  = read_image_label_vector(test_image_file,test_label_file,0,amount)
    for i in range(len(K)):
        rate = run_KNN(train_image_mat,train_label_list,traineva_image_mat,traineva_label_list,5)
    error_rate.append(rate)
    print(error_rate)
    

