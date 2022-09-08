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
df.head(5)


# In[3]:


#Customer id is unique and has no significance on target hence dropping customer id

df.drop('customerID',axis='columns',inplace=True)


# In[4]:


print("Shape of dataset ",df.shape)
print("\n\n Total columns in dataset :\n%s"%str(df.columns))


# In[5]:


print("data types before conversion:  \n%s"%str(df.dtypes))
#To change datatype of TotalCharges from object to numeric one 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')


# In[6]:


print("data typesnversion after co:  \n%s"%str(df.dtypes))


# In[7]:


#describing dataset
print("listed all statistical parameter using Describe() function")
df.describe()


# In[8]:


print("Listed all information using info() function \n\n")
df.info()


# In[9]:


#To check is there are any null values
print(df.isnull().any())


# In[10]:


print('# No of null values in TotalCharges:-',df["TotalCharges"].isnull().sum())


# In[11]:


df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]


# In[12]:


#Dropping NAN values record as their tenure is also 0
df= df.dropna() 


# In[13]:


df.info()


# In[14]:


#To find Outliers
import warnings
warnings.filterwarnings('ignore')
count=1
for i in df.columns:
    if(df[i].dtype=="int64" or df[i].dtype=="float64"):
        plt.figure(figsize=(10,10),tight_layout=True)
        plt.subplot(3,4,count,frameon=False)
        sns.boxplot(df[i])
plt.show()


# In[15]:


#Converting Categorical data into numeric one 
def categoricalDataToNumric():
    global Numeric_Dict
    Numeric_Dict={}
    for i in df.columns:
        if(df[i].dtype=='object'):
            count=1
            empty_coll={}
            temp_list=list(df[i].unique())
            temp_list.sort()
            for j in temp_list:
                empty_coll.update({j:count})
                count+=1
            Numeric_Dict.update({i:empty_coll})


# In[16]:


categoricalDataToNumric()
print("Directory used in converting\n\n")
print(Numeric_Dict)
dset_numerical = df.replace(Numeric_Dict)
dset_numerical.head()


# In[17]:


dset_numerical.dtypes


# In[18]:


#To display total count of datset 
for i in df.columns:
    if(df[i].dtype=='object'):
        plt.figure(figsize=(10,4))
        sns.histplot(df[i])
        plt.xticks(rotation=30,
    horizontalalignment='right',
    fontweight='light',
                fontsize='x-large')


# In[19]:



sns.pairplot(df)


# # Observation
# 
# #1) As monthly charges are increasing total charges are also increasing 

# In[20]:


#bivariate analysis of data
for i in df.columns:
    if(i!='Churn'):
        plt.figure(figsize = (20,4))
        sns.countplot(x=i,hue='Churn',data=df)
        plt.xticks(rotation=30,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large')


# # Observations
# 
# #1)For gender there is not much difference 
# #2)When there is partner chrurning rate is less
# #3)Having dependants also showing less churning rate
# #4)Less the contract more the churning rate as no contract involved
# #5)For the Payment method "ELectronic check" there is high churn rate
# #6)No Technical support provided then churn rate is high
# 
# 

# In[21]:


plt.figure(figsize=(16,6))
heatmap=sns.heatmap(dset_numerical.corr(),vmin=-1,vmax=1,annot=True,cmap="RdBu")
heatmap.set_title('Correlation HeatMap',fontdict={'fontsize':16},pad=12);


# In[22]:


#Splitting entire dataset in test dataset and train dataset
x= dset_numerical.iloc[:,:-1]
y= dset_numerical.iloc[:,-1]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,shuffle=True,random_state=0)
x_train


# In[23]:


y_train


# In[24]:


report= sv.compare([x_train,'Train'],[x_test,'Test'])
report.show_html('compare.html',open_browser=True)


# In[91]:


df= pd.read_csv('C:/Users/LENOVO/Downloads/telco-customer-churn.csv')
#Splitting entire dataset in test dataset and train dataset
x= dset_numerical.iloc[:,:-1]
y= dset_numerical.iloc[:,-1]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,shuffle=True,random_state=0)

def split_data(df):
    df=df.rename(columns={'Churn_yes':'labels})
    dfx= df.drop(['labels'],axis=1)
    dfy=df['labels]

print("Total training set -",len(x_train))
print("Total test set -",len(x_test))


# In[77]:


df= pd.concat([x_train,y_train],axis=1)
df=df.rename(columns={'labels':'target'})


# Logistic Regression With Imbalanced Data

# In[78]:


#Analyzing data using Logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

acc=accuracy_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)

results1=pd.DataFrame([['Logistic Regression',acc,prec,rec,f1]], columns=['Model','Accuracy', 'Precision','Recall','F1 Score'])
results1=results1.sort_values(["Precision","Recall"],ascending=False)
print(results1)


# In[79]:


# Naive Bayes with Imbalanced dataset
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

model_result = pd.DataFrame([['Naive Bayes',acc,prec,rec,f1]], columns=['Model','Accuracy', 'Precision','Recall','F1 Score'])
results1=results1.append(model_result,ignore_index=True)
print(results1)


# # Parameters are affecting score
# classifier = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
#        gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
#        min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
#        objective='binary:logistic', reg_alpha=0, reg_lambda=1,
#        scale_pos_weight=1, seed=0, silent=True, subsample=1)

# In[80]:


#XGBoost model with Imbalanced dataset
from xgboost import XGBClassifier
classifier = XGBClassifier(eval_metric='mlogloss',
                           base_score=0.5,
                           colsample_bylevel=1,
                           colsample_bytree=0)

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

model_result1 = pd.DataFrame([['XGBoost Classifier',acc,prec,rec,f1]], columns=['Model','Accuracy', 'Precision','Recall','F1 Score'])
results1=results1.append(model_result1,ignore_index=True)
print(results1)


# In[81]:


#Random Forest model with Imbalanced Dataset

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 78, criterion='entropy', random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

model_result3 = pd.DataFrame([['Random Forest',acc,prec,rec,f1]], columns=['Model','Accuracy', 'Precision','Recall','F1 Score'])
results1=results1.append(model_result3,ignore_index=True)
print(results1)


# Observation: related to imbalanced data
# 1) Recall of XGBoost classifier is higher
# 
#                 Model  Accuracy  Precision    Recall  F1 Score
# 0  Logistic Regression  0.805970   0.846782  0.899807  0.872489
# 1          Naive Bayes  0.749822   0.888009  0.756262  0.816857
# 2   XGBoost Classifier  0.801706   0.836735  0.908478  0.871132
# 3        Random Forest  0.794598   0.834674  0.899807  0.866018

# In[82]:


#Balance data using SMOTE technique

from imblearn.over_sampling import SMOTE
from sklearn.neighbors import DistanceMetric
smote= SMOTE()


# In[83]:


X_train_smote, Y_train_smote = smote.fit_resample(x_train,y_train)


# In[84]:


from collections import Counter
print("Before Smote",Counter(y_train))
print("After Smote",Counter(Y_train_smote))


# In[95]:


# Logistic regression using Balanced Data
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_smote,Y_train_smote)

y_pred=classifier.predict(x_test)

acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

model_result = pd.DataFrame([['Logistic Regression after smote', acc, prec, rec, f1]], columns=['Model','Accuracy', 'Precision','Recall','F1 Score'])
results1=results1.append(model_result,ignore_index=True)
print(results1)


# In[86]:


#Naive bayes with Balanced dataset 

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train_smote,Y_train_smote)

y_pred=classifier.predict(x_test)

acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

model_result2 = pd.DataFrame([['Naive Bayes after smote',acc,prec,rec,f1]], columns=['Model','Accuracy', 'Precision','Recall','F1 Score'])
results1=results1.append(model_result2,ignore_index=True)
print(results1)


# In[87]:


#XGBoost classifier with Balanced dataset  

from xgboost import XGBClassifier
classifier = XGBClassifier(eval_metric='mlogloss',
                           scale_pos_weight=1, 
                           learning_rate=0.01,
                           base_score=0.5,colsample_bylevel=1,
                           colsample_bytree=0)
classifier.fit(X_train_smote,Y_train_smote)

y_pred=classifier.predict(x_test)

acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

model_result3 = pd.DataFrame([['XGBoost Classifier after smote',acc,prec,rec,f1]], columns=['Model','Accuracy', 'Precision','Recall','F1 Score'])
results1=results1.append(model_result3,ignore_index=True)
print(results1)


# In[88]:


#RandomForest on Balanced data

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 72, criterion='entropy', random_state=0)
classifier.fit(X_train_smote,Y_train_smote)

y_pred=classifier.predict(x_test)

acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

model_result3 = pd.DataFrame([['Random Forest after smote',acc,prec,rec,f1]], columns=['Model','Accuracy', 'Precision','Recall','F1 Score'])
results1=results1.append(model_result3,ignore_index=True)
print(results1)


#                          Model      Accuracy  Precision    Recall  F1 Score
# Logistic Regression                 0.805970   0.846782  0.899807  0.872489
#     Naive Bayes                     0.749822   0.888009  0.756262  0.816857
#       XGBoost Classifier            0.801706   0.836735  0.908478  0.871132
#  Random Forest                      0.794598   0.834674  0.899807  0.866018
# Logistic Regression after smote     0.863326   0.888644  0.776493  0.828792
# Naive Bayes after smote             0.740583   0.883694  0.796628  0.809399
# XGBoost Classifier after smote      0.731343   0.898551  0.916763  0.79742             
# Random Forest after smote           0.792466   0.859345  0.859345  0.859345
# 

# ##### HYPERTUNING

# In[38]:


from xgboost import XGBClassifier
XGBClassifier()


# In[94]:


#Hypertuning XGBoost classifier

#"subsample" is the fraction of the training samples (randomly selected) that will be used to train each tree.
#"colsample_by_tree" is the fraction of features (randomly selected) that will be used to train each tree.
#"colsample_bylevel" is the fraction of features (randomly selected) that will be used in each node to train each tree.
from xgboost import XGBClassifier
classifier = XGBClassifier(eval_metric='mlogloss', 
                           scale_pos_weight=1, 
                           learning_rate=0.5,
                           colsample_bytree=0.4,
                           subsample=0.8)
classifier.fit(X_train_smote,Y_train_smote)


y_pred=classifier.predict(x_test)

acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

result = pd.DataFrame([['XGBoost Hypertuned results',acc,prec,0.rec,f1]], columns=['Model','Accuracy', 'Precision','Recall','F1 Score'])

print(result)


# In[74]:


from sklearn.model_selection import RandomizedSearchCV
# no of trees to be generate
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 10)]
# at splitting points below features will be consider
max_features = ['auto', 'sqrt']
# How many levels it will go down
max_depth = [int(x) for x in np.linspace(20, 240, num = 11)]
max_depth.append(None)
# min required sample to split the data
min_samples_split = [2, 5, 10]
# min leaf node sample
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[90]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train, y_train)


# In[75]:


rf_random.best_params_


# In[ ]:


Observation 

Imbalance data 
1) XG boost is the best algorithm
        XGBoost Classifier  0.801706   0.836735  0.908478  0.871132
2) Random Forest is worst algorithm
Balance Data
1) XGBoost Classifier after smote  0.734897   0.901086  0.719653  0.800214

HyperTuning
1) With Hypertuning able to increse recall rate of XGboost model

