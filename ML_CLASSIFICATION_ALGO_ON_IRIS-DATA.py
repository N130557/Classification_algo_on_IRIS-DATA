
# coding: utf-8

# # step1 : Check all the versions as per the Readme.txt

# In[5]:


import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn


# In[ ]:


print("python: {}".format(sys.version))
print("scipy: {}".format(scipy.__version__))
print("numpy: {}".format(numpy.__version__)
print("matplotlib: {}".format(matplotlib.__version__))
print("pandas:{}".format(pandas.__version__))
print("sklearn: {}".format(sklearn.__version__))


# # step2 : import necessary modules from above packages

# In[ ]:


from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# # step3 : Import Data-Set

# In[ ]:


url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
datasets=pandas.read_csv(url,names=names)


#      sepal-length  sepal-width  petal-length  petal-width           class
# 0             5.1          3.5           1.4          0.2     Iris-setosa
# 1             4.9          3.0           1.4          0.2     Iris-setosa
# 2             4.7          3.2           1.3          0.2     Iris-setosa
# 3             4.6          3.1           1.5          0.2     Iris-setosa
# 4             5.0          3.6           1.4          0.2     Iris-setosa
# 5             5.4          3.9           1.7          0.4     Iris-setosa
# 6             4.6          3.4           1.4          0.3     Iris-setosa
# 7             5.0          3.4           1.5          0.2     Iris-setosa
# 8             4.4          2.9           1.4          0.2     Iris-setosa
# 9             4.9          3.1           1.5          0.1     Iris-setosa
# 10            5.4          3.7           1.5          0.2     Iris-setosa
# 11            4.8          3.4           1.6          0.2     Iris-setosa
# 12            4.8          3.0           1.4          0.1     Iris-setosa
# 13            4.3          3.0           1.1          0.1     Iris-setosa
# 14            5.8          4.0           1.2          0.2     Iris-setosa
# 15            5.7          4.4           1.5          0.4     Iris-setosa
# 16            5.4          3.9           1.3          0.4     Iris-setosa
# 17            5.1          3.5           1.4          0.3     Iris-setosa
# 18            5.7          3.8           1.7          0.3     Iris-setosa
# 19            5.1          3.8           1.5          0.3     Iris-setosa
# 20            5.4          3.4           1.7          0.2     Iris-setosa
# 21            5.1          3.7           1.5          0.4     Iris-setosa
# 22            4.6          3.6           1.0          0.2     Iris-setosa
# 23            5.1          3.3           1.7          0.5     Iris-setosa
# 24            4.8          3.4           1.9          0.2     Iris-setosa
# 25            5.0          3.0           1.6          0.2     Iris-setosa
# 26            5.0          3.4           1.6          0.4     Iris-setosa
# 27            5.2          3.5           1.5          0.2     Iris-setosa
# 28            5.2          3.4           1.4          0.2     Iris-setosa
# 29            4.7          3.2           1.6          0.2     Iris-setosa
# ..            ...          ...           ...          ...             ...
# 120           6.9          3.2           5.7          2.3  Iris-virginica
# 121           5.6          2.8           4.9          2.0  Iris-virginica
# 122           7.7          2.8           6.7          2.0  Iris-virginica
# 123           6.3          2.7           4.9          1.8  Iris-virginica
# 124           6.7          3.3           5.7          2.1  Iris-virginica
# 125           7.2          3.2           6.0          1.8  Iris-virginica
# 126           6.2          2.8           4.8          1.8  Iris-virginica
# 127           6.1          3.0           4.9          1.8  Iris-virginica
# 128           6.4          2.8           5.6          2.1  Iris-virginica
# 129           7.2          3.0           5.8          1.6  Iris-virginica
# 130           7.4          2.8           6.1          1.9  Iris-virginica
# 131           7.9          3.8           6.4          2.0  Iris-virginica
# 132           6.4          2.8           5.6          2.2  Iris-virginica
# 133           6.3          2.8           5.1          1.5  Iris-virginica
# 134           6.1          2.6           5.6          1.4  Iris-virginica
# 135           7.7          3.0           6.1          2.3  Iris-virginica
# 136           6.3          3.4           5.6          2.4  Iris-virginica
# 137           6.4          3.1           5.5          1.8  Iris-virginica
# 138           6.0          3.0           4.8          1.8  Iris-virginica
# 139           6.9          3.1           5.4          2.1  Iris-virginica
# 140           6.7          3.1           5.6          2.4  Iris-virginica
# 141           6.9          3.1           5.1          2.3  Iris-virginica
# 142           5.8          2.7           5.1          1.9  Iris-virginica
# 143           6.8          3.2           5.9          2.3  Iris-virginica
# 144           6.7          3.3           5.7          2.5  Iris-virginica
# 145           6.7          3.0           5.2          2.3  Iris-virginica
# 146           6.3          2.5           5.0          1.9  Iris-virginica
# 147           6.5          3.0           5.2          2.0  Iris-virginica
# 148           6.2          3.4           5.4          2.3  Iris-virginica
# 149           5.9          3.0           5.1          1.8  Iris-virginica
# 
# [150 rows x 5 columns]
# 
# datasets.shape

# # step4 : operations on Dataset

# In[ ]:


print(datasets.shape)


# output : (150, 5)

# In[ ]:


print(datasets.info())


# output :
# class 'pandas.core.frame.DataFrame'>
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 5 columns):
# sepal-length    150 non-null float64
# sepal-width     150 non-null float64
# petal-length    150 non-null float64
# petal-width     150 non-null float64
# class           150 non-null object
# dtypes: float64(4), object(1)
# memory usage: 5.9+ KB
# None

# In[ ]:


print(datasets.describe())


#        sepal-length  sepal-width  petal-length  petal-width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.054000      3.758667     1.198667
# std        0.828066     0.433594      1.764420     0.763161
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.800000     3.000000      4.350000     1.300000
# 75%        6.400000     3.300000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000

# In[ ]:


print(datasets.groupby('class').size())


# output : prints the how many types of classes are existed in out dataset.
# 
# class
# Iris-setosa        50
# Iris-versicolor    50
# Iris-virginica     50
# dtype: int64
# 
# 

# # step 5 : visualization of Dataset

# In[ ]:


datasets.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()


# # step 6 : splitting Dataset into Training and Test set.

# In[ ]:


array=datasets.values
X=array[:,0:4]
y=array[:,4]
validation_size=0.20
seed=6
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,y,test_size=validation_size,random_state=seed)


# In[ ]:


seed=6
scoring='accuracy'


# #  step 7 : import classification algo from Scikit-Learn library

# In[ ]:


models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))


# names=[]
# for name,model in models:
# 	kfold=model_selection.KFold(n_splits=10,random_state=seed)
# 	cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
# 	results.append(name)
# 	names.append(name)
# 	msg="%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
# 	print(msg)
#     

# # step 8 : each classification model accuracy

# output : 
# LR: 0.941667 (0.038188)
# LDA: 0.975000 (0.038188)
# KNN: 0.958333 (0.055902)
# CART: 0.933333 (0.050000)
# NB: 0.966667 (0.055277)
# SVM: 0.966667 (0.055277)
# 
