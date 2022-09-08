#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import recall_score

df= pd.read_csv('C:/Users/LENOVO/Desktop/Be4 Pace/MS/676 Algo for data science/Project/winequality-white.csv')


# In[46]:


df


# In[2]:


X= df.iloc[:,0:10]
Y= df.iloc[:,11]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
classifier = KNeighborsClassifier(n_neighbors = 3,p=2,metric='minkowski')


# In[3]:


classifier.fit(x_train,y_train)


# In[4]:


y_pred= classifier.predict(x_test)


# In[51]:


y_pred


# In[8]:


import math

def get_distance(test1,test2):
  
    euclidean_list_complete = []
    outer_length1 = len(test1)
    outer_length2 = len(test2)
    inner_length = len(test2[0])
    
    for i in range(outer_length2):
        euclidean_list = []
        for j in range(outer_length1):
            euclidean = 0
            for k in range(inner_length):
                euclidean += pow(float(test2[i][k]) - float(test1[j][k]), 2)
    
            euclidean_list.append(math.sqrt(euclidean))
    
        euclidean_list.sort(reverse = True)
        euclidean_list_complete.append(euclidean_list)
    
    return euclidean_list_complete

traintt=np.array([[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4],[7.4, 0.6, 0, 1.2, 0.076, 1, 24, 0.9978, 3.51, 0.56, 9.4]])
Testtt= np.array([[7.3, 0.65, 0,1.2, 0.065, 15, 21, 0.9946, 3.39, 0.47, 10]])


print(
        'Euclidean Distance between two matrix :\n',
        get_distance(traintt,Testtt
        )
    )


# In[52]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print(result)

result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[87]:


#problem 1

k = 1.0
while k < 3.0:
    classifier = KNeighborsClassifier(n_neighbors = 1,p=k,metric='minkowski')
    classifier.fit(x_train,y_train)
    y_pred= classifier.predict(x_test)
    result2 = accuracy_score(y_test,y_pred)
    print("Accuracy:",result2," value of p",k)
    k += 0.1


# Problem 2:  Find the optimal k of the kNN classifier. Use L2.

# In[127]:


k = 1
while k <= 11 :
    classifier = KNeighborsClassifier(n_neighbors = k,p=2,metric='euclidean')
    classifier.fit(x_train,y_train)
    y_pred= classifier.predict(x_test)
    result2 = accuracy_score(y_test,y_pred)
    recall= recall_score(y_test, y_pred, average='macro')
    print("Accuracy:",result2,"Recall := ",recall," K value :=",k)
    k += 2
    
print("optimal K value is: 1")


# In[74]:


import math

# Return the square root of different numbers
print(math.sqrt(len(y_test)))


# In[103]:


k = 0.5
while k <= 2.5:
    knn = KNeighborsClassifier(n_neighbors =5,metric='weighted', p=99, 
                           metric_params={'w': np.random.random(x_train.shape[1])})
    classifier.fit(x_train,y_train)
    n_neigh=5
    y_pred= classifier.predict(x_test)
    result2 = accuracy_score(y_test,y_pred)
    recall= recall_score(y_test, y_pred, average='macro')
    print("Accuracy:",result2,", P Value",k)
    k += 0.5


# In[120]:


#Normalization using min max 

from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

data = df.values[:, :-1]

# convert the array back to a dataframe
df2 = DataFrame(data)

df_norm = (df2-df2.min())/ (df2.max() - df2.min())
# summarize

print("norm", df_norm)


# In[125]:


#Q 4. Find the p value that gives the least error rate of the nearest neighbor classifier when
# Minkowski Lp is used. Try p = 0.5 ∼ 2.5 incremented by 0.1.

x= df_norm.iloc[:, :-1]
y= df_norm.iloc[:, :1]
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2)


k = 1.0
while k < 3.0:
    classifier = KNeighborsClassifier(n_neighbors = 5,p=k,metric='minkowski')
    classifier.fit(x_train,y_train)
    y_pred= classifier.predict(x_test)
    result2 = accuracy_score(y_test,y_pred)
    print("Accuracy:",result2," value of p",k)
    k += 0.1


# In[130]:


#Q 5. Find the optimal k of the kNN classifier. Use L2.

k = 1
while k <= 11 :
    classifier = KNeighborsClassifier(n_neighbors = k,p=2,metric='euclidean')
    classifier.fit(x_train,y_train)
    y_pred= classifier.predict(x_test)
    result2 = accuracy_score(y_test,y_pred)
    recall= recall_score(y_test, y_pred, average='macro')
    print("Accuracy:",result2,"Recall := ",recall," K value :=",k)
    k += 2
    
print("optimal K value is: 1")


# In[131]:


#Q 6. Find the optimal p value for the distance wi =1/d(ri, q)pweighted (k = 5)-NN classifier.Try at least p = 0.5, 1, 1.5, 2, 2.5. Use L2.

k = 0.5
while k <= 2.5:
    knn = KNeighborsClassifier(n_neighbors =5,metric='weighted', p=99, 
                           metric_params={'w': np.random.random(x_train.shape[1])})
    classifier.fit(x_train,y_train)
    n_neigh=5
    y_pred= classifier.predict(x_test)
    result2 = accuracy_score(y_test,y_pred)
    recall= recall_score(y_test, y_pred, average='macro')
    print("Accuracy:",result2,", P Value",k)
    k += 0.5


# In[ ]:


Q 7. Discuss the best result of the above experiments

Observation :
    1) when we increse K value we are getting more accurate data 
    2) The above experiment i have carried on the basis of accuracy only but going forward I will try 
       to ionclude confusion matrix and then will compare on the same
    3) In a distance weighted measure as we are increaing value of p, the less the weight we are getting

