# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties 
from sklearn import svm


data=np.loadtxt('data2.txt')

X=data[:,0:2]
y=data[:,2]

pos = np.where(y == 1)  
neg = np.where(y == 0)  

plt.figure(1,figsize=(8,6))

# clf = svm.SVC(kernel='linear').fit(X, y)
# clf = svm.SVC(kernel='poly',degree=5,gamma=1,coef0=0).fit(X, y)
clf = svm.SVC(kernel='rbf',C=100,gamma=20).fit(X, y)

'''gamma越大，多项式项数越多,导致高方差'''


print u'精准度为： %.2f' % clf.score(X, y)

x1_min, x1_max = X[:, 0].min(), X[:, 0].max()  # 第0列的范围
x2_min, x2_max = X[:, 1].min(), X[:, 1].max() 


x1,x2=np.mgrid[x1_min:x1_max:200j,x2_min:x2_max:200j]
grid_test = np.stack((x1.flat, x2.flat), axis=1)
# print grid_test

grid_test_pre=clf.predict(grid_test)

grid_test_pre=grid_test_pre.reshape(x1.shape)

# print grid_test_pre
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])   
plt.pcolormesh(x1,x2,grid_test_pre,cmap=cm_light)

myfont = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14) 
plt.scatter(X[pos,0],X[pos,1],color="red",label="1",linewidth=3)
plt.scatter(X[neg,0],X[neg,1],color="blue",label="0",linewidth=3)
plt.xlabel(u'Exam1 Score',fontproperties=myfont)
plt.ylabel('Exam2 Score')
plt.legend()

plt.show()
