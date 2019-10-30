#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pa
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pa.read_csv(r'Telecom_churn_data.csv')


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# #### This indicates there are nno null value in the data set so we can proceed with the feature processing

# In[5]:


data.describe()


# ## Data Featuring

# In[6]:


churn = pa.get_dummies(data.Churn, drop_first = True)
data = data.drop('Churn', axis = 1)
#data = pa.concat([data, churn], axis = 1)
#data.head()


# In[7]:


data = pa.concat([data, churn], axis = 1)
data.head()


# In[8]:


data.rename({"Yes" : "churn"}, axis =1, inplace = True)


# In[9]:


data.head()


# In[10]:


x = data.iloc[:,:20]


# In[11]:


y = data.iloc[:,20]


# In[12]:


x.head()


# #### Gender

# In[13]:


gender = pa.get_dummies(x['gender'], drop_first = True)
x = x.drop('gender', axis = 1)
x = pa.concat([x, gender], axis =1)
x.head()


# #### Partner

# In[14]:


partner = pa.get_dummies(x['Partner'], drop_first=True)
x = x.drop('Partner', axis =1)
x = pa.concat([x, partner], axis = 1)
x.head()


# In[15]:


x.rename({"Yes":"partner"}, axis = 1, inplace= True)
x.head()


# #### Processing of Dependents

# In[16]:


dependents = pa.get_dummies(x['Dependents'], drop_first=True)
x = x.drop('Dependents', axis =1)
x = pa.concat([x, dependents], axis = 1)
x.head()


# In[17]:


x.rename({"Yes":"dependents"}, axis = 1, inplace = True)
x.head()


# #### Processing phone service

# In[18]:


phone_service = pa.get_dummies(x['PhoneService'], drop_first=True)
x = x.drop('PhoneService', axis =1)
x = pa.concat([x, phone_service], axis = 1)
x.rename({"Yes":"phone_service"}, axis = 1, inplace = True)


# In[19]:


x.head()


# #### Processing multiple lines

# In[20]:


Multiplelines = pa.get_dummies(x['MultipleLines'], drop_first=True)
x = x.drop('MultipleLines', axis =1)
x = pa.concat([x, Multiplelines], axis = 1)
x.head()


# In[21]:


x.rename({"Yes":"Multiplelines"}, axis = 1, inplace = True)
x.head()


# #### Processing internet services

# In[22]:


#using one hot encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[23]:


x.InternetService = le.fit_transform(x.InternetService)
x.head()


# ##### processing online backupu

# In[24]:


x.OnlineBackup = le.fit_transform(x.OnlineBackup)
x.head()


# ##### processing device protection

# In[25]:


x.DeviceProtection = le.fit_transform(x.DeviceProtection)
x.head()


# ##### processing tech support

# In[26]:


x.TechSupport = le.fit_transform(x.TechSupport)
x.head()


# ##### processing tv streming

# In[27]:


x.StreamingTV = le.fit_transform(x.StreamingTV)
x.head()


# ##### processing online security

# In[28]:


x.OnlineSecurity = le.fit_transform(x.OnlineSecurity)
x.head()


# #### processing streaming movie

# In[29]:


x.StreamingMovies = le.fit_transform(x.StreamingMovies)
x.head()


# ##### processing contract

# In[30]:


x.Contract = le.fit_transform(x.Contract)
x.head()


# ##### processing paperless bill

# In[31]:


x.PaperlessBilling = le.fit_transform(x.PaperlessBilling)
x.head()


# #### processing payment method

# In[32]:


x.PaymentMethod = le.fit_transform(x.PaymentMethod)
x.head()


# In[33]:


x = x.drop('customerID', axis=1)


# In[34]:


x.TotalCharges = pa.to_numeric(x.TotalCharges, errors = 'coerce')
x.TotalCharges.dtype


# #### split dataset into train and test dataset

# In[35]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=0)


# In[36]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ## fit modol

# In[37]:


from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(x_train, y_train)


# In[38]:


y_pred = classifier.predict(x_test)


# ### make confusion matrics

# In[39]:


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)
cm


# #### The accuracy calculation

# In[40]:


(938+184)/(938+184+184+103)


# ## This indicates model has accuracy of 79.6%

# ### Tuning the xgboost model using hyper parameter

# In[41]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[42]:


params = {
    "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth"        : [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight" : [1, 3, 5, 7],
    "gamma"            : [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree" : [0.3, 0.4, 0.5, 0.7]
}


# In[49]:


import xgboost


# In[50]:


classifier1 = xgboost.XGBClassifier()


# In[51]:


random_search = RandomizedSearchCV(classifier1, param_distributions=params, n_iter=5, scoring = "roc_auc", n_jobs =-1, cv = 5, verbose = 3)


# In[52]:


random_search.fit(x, y)


# In[53]:


random_search.best_estimator_


# In[54]:


random_search.best_params_


# In[55]:


classfier = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0.0,
              learning_rate=0.15, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)


# In[58]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(classifier1, x, y, cv = 10)


# In[59]:


score


# In[60]:


score.mean()


# In[ ]:




