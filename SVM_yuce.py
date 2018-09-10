# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties 
from sklearn import svm


data=np.loadtxt('yucedata1.txt')

X=data[:,0]
y=data[:,1]

plt.figure(1,figsize=(8,6))
myfont = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14) 
plt.scatter(X,y,color="red",label="ini_data",linewidth=3)
plt.xlabel(u'Exam1 Score',fontproperties=myfont)
plt.ylabel('Exam2 Score')
plt.legend()

# plt.show()
X=X.reshape(-1,1)
print X
clf = svm.SVR(kernel='linear').fit(X, y)
# clf = svm.SVC(kernel='poly',degree=5,gamma=1,coef0=0).fit(X, y)
# clf = svm.SVR(kernel='rbf',C=100,gamma=20).fit(X, y)

'''gamma越大，多项式项数越多,导致高方差'''


# print u'精准度为： %.2f' % clf.score(X, y)

X1=np.linspace(0,25,100).reshape(-1,1)

y1=clf.predict(X1)

plt.plot(X1,y1,color="orange",label="Fitting Line",linewidth=2) 


plt.show()
