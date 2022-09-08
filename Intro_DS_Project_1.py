#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Data Processing Libraries
import numpy as np 
import pandas as pd

#Data Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv

#to display inline 
get_ipython().run_line_magic('matplotlib', 'inline')

#to split dataset into train and test data set
from sklearn.model_selection import train_test_split


# In[2]:


#read file from location and loading it for processing 
df= pd.read_csv('C:/Users/LENOVO/Downloads/telco-customer-churn.csv')


# In[3]:


#loading top 5 and bottom 5 rows for analysis
df #top 5 rows and bottom 5 rows


# In[4]:


#Basic understanding of data set
df.shape #output = (rows,columns)


# In[5]:


df.info()


# In[6]:


#Customer id is unique and has no significance on target label hence dropping customer id

df.drop('customerID',axis='columns',inplace=True)


# In[7]:


df.info()


# In[8]:


#To change datatype of TotalCharges from object to numeric one 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')


# In[9]:


df.info()


# In[10]:


#To check is there are any null values
print(df.isnull().any())


# In[11]:


print('# No of null values in TotalCharges:-',df["TotalCharges"].isnull().sum())


# In[12]:


df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]


# In[13]:


#Dropping NAN values record as their tenure is also 0
df= df.dropna() 


# In[14]:


df.info()


# In[15]:


df.shape


# In[16]:


df.dtypes


# # In given dataset now we have 16 categorical columns whaeras 4 numerical columns

# In[17]:


#to get values of categorical columns
def print_col_values(df):
    for column in df:
        if df[column].dtypes=='object':
            print(f'{column}: {df[column].unique()}')


# In[18]:


print_col_values(df)


# In[19]:


df.replace('No internet service','No',inplace=True)
df.replace('No phone service','No',inplace=True)


# In[20]:


print_col_values(df)


# In[21]:


binary_Columns=['SeniorCitizen','Partner','PaperlessBilling','PhoneService','MultipleLines','OnlineSecurity','OnlineSecurity',
               'OnlineBackup','Dependents','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','StreamingMovies','Churn']

for col in binary_Columns:
    df[col].replace({'Yes':1,'No':0},inplace=True)


# In[22]:


df['gender'].replace({'Female':1, 'Male':0})


# In[23]:


df.gender.unique()


# In[24]:


for col in df:
    print(f'{col}:{df[col].unique()}')


# In[25]:


df.dtypes


# In[26]:


InternetService = {
    'DSL' : 11,
    'Fiber optic' : 22,
    'No' : 33
}
Contract={
    'Month-to-month' : 44,
    'One year' :55,
    'Two year':66
}
PaymentMethod={
    'Electronic check':111,
    'Mailed check':222,
    'Bank transfer (automatic)':333,
    'Credit card (automatic)':444
}
Internet_service = df['InternetService'].map(InternetService)
Contract_data = df['Contract'].map(Contract)
Payment_Method= df['PaymentMethod'].map(PaymentMethod)


# In[27]:


new_data = df.copy()
new_data['InternetService'] = Internet_service
new_data['Contract'] = Contract_data 
new_data['PaymentMethod'] = Payment_Method
print(new_data)


# In[28]:


new_data.dtypes


# In[29]:


new_data['gender'].replace({'Female':1, 'Male':0},inplace=True)


# In[30]:


new_data.gender.unique()


# In[31]:


#converted all data to numeric datatype
new_data.dtypes


# In[32]:


new_data.describe().T


# In[33]:


plt.figure(figsize = (20,40))
sns.boxplot(x="value",y="variable",data=pd.melt(new_data))
plt.show()


# In[44]:


#Getting the list of all columns
columns_list=list(binary_Columns)

#converting columns to display into matrix format
columns_array = np.array(columns_list)
columns_array = np.reshape(columns_array,(5,3))


# In[45]:


rows = 5; columns = 3
f,axes=plt.subplots(rows,columns,figsize=(30,30))
print("Analysis of each feature with Bar chart")
for row in range(rows):
    for column in range(columns):
        sns.countplot(df[columns_array[row][column]],palette="Set1",ax=axes[row,column])


# In[ ]:




