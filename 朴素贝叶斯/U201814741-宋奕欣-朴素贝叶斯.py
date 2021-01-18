import numpy as np
import re
import random
import jieba
import math
 
"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表
"""
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)
 

 


"""
函数说明:接收一个大字符串并将其解析为字符串列表
"""
def textParse(bigString):  # 将字符串转换为字符列表
    listOfTokens = re.split(r'\n', bigString)  # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    #return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 除了单个字母，例如大写的I，其它单词变成小写
    return listOfTokens
"""
函数说明:测试朴素贝叶斯分类器，使用朴素贝叶斯进行交叉验证
"""
def spamTest():
    docList = []
    classList = []
    fullText = []
    #stopList = textParse(open('stop.txt','r',encoding='UTF-8').read());
    #stopList.append('\n')
    #stopList.append('\t')
    #stopList.append('\u3000')
    stopList = []
    #print(stopList)
    indexOfSet = open('newindex','r').read();
    indexOfSet = indexOfSet.split('\n')
    #seglist = jieba.cut(datas,cut_all=False)
    lengthOfFile = len(indexOfSet)
    #j = 0
    for i in indexOfSet:
        filePath = i[4:]
        filePath = "".join(filePath)
        classType = i[0:4]
        datas = open(filePath,'r').read();
        datas = re.sub("[A-Za-z0-9\!\%\[\]\,\。\+\-\_\.]","",datas)
        seglist = jieba.cut(datas,cut_all=False)
        nowList = ""
        for seg in seglist:
            if seg not in stopList:
                #print(seg)
                nowList += seg
        seglist = jieba.lcut(nowList,cut_all=False)
        #print(seglist)
        docList.append(seglist)
        if(classType=='ham '):
            classList.append(0)
        else:
            classList.append(1)
        #j += 1
        #print(j)
    print('预处理结束')
    vocabList = createVocabList(docList)  # 创建词汇表，不重复
    print('词汇表创建完成')
    #print(vocabList)
    trainingSet = list(range(15320,lengthOfFile))
    testSet = list(range(0,15320))  # 创建存储训练集的索引值的列表和测试集的索引值的列表
    
    #print(testSet)
    #print(trainingSet)
    frequency1 = {}
    frequency0 = {}
    p1num = 0
    p0num = 0
    p1wordAllNum = 2
    p0wordAllNum = 2
    p1Vect = []
    p1index = []
    p0index = []
    p0Vect = []
    n = 0
    for docIndex in trainingSet:
        #print(docIndex)
        if(classList[docIndex] == 1):#是spam
            p1num += 1
            for word in docList[docIndex]:
                p1wordAllNum += 1
                if word not in frequency1:
                    frequency1[word] = 1
                else:
                    frequency1[word] += 1
        else:
            p0num += 1
            for word in docList[docIndex]:
                p0wordAllNum += 1
                if word not in frequency0:
                    frequency0[word] = 1
                else:
                    frequency0[word] += 1
    print('the spam num is',p1num)
    #print(frequency1)
    #print(p1wordAllNum)
    #print(p0wordAllNum)
    pAbusive = p1num / (float)(p1num + p0num)
    for key in frequency1:
        #print(key)
        gailv = math.log((frequency1[key]+1)/p1wordAllNum)
        #print(key,':',gailv,':1:',frequency1[key]+1)
        p1index.append(key)
        p1Vect.append(gailv)
    for key in frequency0:
        gailv = math.log((frequency0[key]+1)/p0wordAllNum)
        #print(key,':',gailv,':1:',frequency0[key]+1)
        p0index.append(key)
        p0Vect.append(gailv)
    print(p1Vect)
    falseNum = 0
    spamErr = 0
    hamErr = 0
    print(frequency1)
    print(frequency0)
    lowest = min(min(p1Vect),min(p0Vect))
    print(lowest)
    for docIndex in testSet:
        testDict = {}
        ans1 = math.log(pAbusive)
        ans0 = math.log(1-pAbusive)
        for word in docList[docIndex]:
            if word not in testDict:
                testDict[word] = 1
                if word in frequency1:
                    ans1 = ans1 + p1Vect[p1index.index(word)]
                    #print(p1Vect[p1index.index(word)],':',word)
                else:
                    ans1 = ans1 + lowest
                if word in frequency0:
                    ans0 = ans0 + p0Vect[p0index.index(word)]
                else:
                    ans0 = ans0 + lowest
        #print(ans1,':',ans0,':',classList[docIndex])
        #if(ans1 > ans0):
            #print('预测的结果为:1,实际结果为:',classList[docIndex])
        #else:
            #print('预测的结果为:0,实际结果为:',classList[docIndex])
        if ((ans1 > ans0) and classList[docIndex] != 1):
            falseNum += 1
            spamErr += 1
        elif ((ans1 < ans0) and classList[docIndex] != 0):
            falseNum += 1
            hamErr += 1
        
    print('错误率：%.2f%%' % (float(falseNum) / len(testSet) * 100))
    print('把垃圾邮件误判为正常邮件的数量为',spamErr)
    print('把正常邮件误判为垃圾邮件的数量为',hamErr)
    
 
if __name__ == '__main__':
    spamTest()
    
