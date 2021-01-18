import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

testx = np.array([1,5,10,20,50,100,120])
testy = np.array([3.6,3.35,3.64,4.07,4.96,5.97,6.29])
testxnew = np.arange(1,120,4)
funx = interpolate.interp1d(testx,testy,kind = 'cubic')
ynew = funx(testxnew)
plt.plot(testxnew,ynew,'k-',label="test")
plt.scatter(testx,testy,)

for i in zip(testx,testy):
    plt.text(i[0],i[1],str(i[1]),ha='right',va='bottom')

evax = np.array([1,5,10,20,50,100,120])
evay = np.array([3.34,3.28,3.3,3.88,4.76,5.75,6.08])
evaxnew = np.arange(1,120,4)
evafunx = interpolate.interp1d(evax,evay,kind='slinear')
evaynew = evafunx(evaxnew)
plt.plot(evaxnew,evaynew,'k--',label="evaluate")
plt.scatter(evax,evay)
for i in zip(evax,evay):
    plt.text(i[0],i[1],str(i[1]),ha='right',va='bottom')
plt.xlabel('K的选取')
plt.ylabel("错误率/%")
plt.title('KNN识别MNIST错误率')
plt.legend()
plt.show()
