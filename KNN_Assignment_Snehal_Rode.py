#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


# In[13]:


columnheader = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']


# In[14]:


dataset = pd.read_csv(path, names = columnheader)
dataset.head()


# In[5]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)


# In[8]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[9]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(X_train, y_train)


# In[10]:


y_pred = classifier.predict(X_test)


# In[ ]:





# In[ ]:





# In[11]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[15]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[16]:


print(error_rate)


# In[20]:


# applying groupby() function to
# group the data on team value.
gk = dataset.groupby('Class')
  
# Let's print the first entries
# in all the groups formed.
gk.first()


# In[21]:


groupby = gk.get_group('Iris-setosa')
print(groupby)


# In[23]:


groupby["sepal-length"]


# In[24]:


my_max = groupby.idmax   # Maximum in column
print(my_max)


# In[29]:


gf_max = groupby.max()


# In[30]:


gf_min = groupby.min()


# In[27]:


first_ele =[5.1,3.5,1.4,0.2]
print(first_ele)


# In[34]:


gf_max= groupby.max()
gf_min= groupby.min()


# In[35]:


print(gf_max,gf_min)


# In[37]:


gf=groupby.drop('Class',axis=1)


# In[38]:


print(gf)


# In[40]:


result = (first_ele - gf_min)


# In[ ]:




