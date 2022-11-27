#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import random
# data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
# data modeling
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import regularizers
# evaluation
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pickle
# hide warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# In[2]:


df = pd.read_csv("Life Expectancy Data_HV22.csv")


# In[3]:


df.columns = df.columns.str.strip()


# In[5]:


col_null = df.columns[df.isna().any()].tolist()
for i in col_null:
    null = df[i].isna().sum()
    null = str(null)
    print("The column {} has {} null values".format(i, null))


# In[6]:


for i in col_null:
    df[i].fillna((df[i].mean()), inplace=True)


# In[7]:


class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns # array of column names to encode


    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self


    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output


    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)


    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output


# In[8]:


cat_features = [feature for feature in df.columns if df[feature].dtype in ['object', 'bool_']]
multi = MultiColumnLabelEncoder(columns=cat_features)
df = multi.fit_transform(df)


# In[17]:


dump(multi, open('encoder.pkl', 'wb'))


# In[14]:


df_min_max_scaled = df.copy()

scaler = MinMaxScaler()
scaler.fit(df_min_max_scaled)
# apply normalization techniques
df_min_max_scaled = scaler.transform(df_min_max_scaled)

# view normalized data
print(df_min_max_scaled)


# In[10]:


df_min_max_scaled = pd.DataFrame(data = df_min_max_scaled, columns = df.columns)


# In[15]:


dump(scaler, open('scaler.pkl', 'wb'))


# In[11]:


X = df_min_max_scaled[['Income composition of resources','Country','Adult Mortality','BMI','HIV/AIDS','Schooling']]
y = df_min_max_scaled['Life expectancy']

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.4, random_state=49)

X_train.head()

model =XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("XGBRegressor")
print("rmse: ",np.sqrt(mse))
print("r2 score: ",r2)


# In[12]:


from pickle import dump


# In[13]:


dump(model, open('model.pkl', 'wb'))


# In[ ]:




