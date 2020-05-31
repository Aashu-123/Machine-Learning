
# coding: utf-8

# In[23]:


# Importing all the necessary libraries
import pandas as pd
import numpy as np
from sklearn import linear_model


# In[22]:


# Reading the file with Pandas
df = pd.read_csv(r'C:\Users\Desktop\areas_multiplevariables.csv')


# In[24]:


# Look at the file which is read
df


# In[26]:


#Data Preprocessing: Fill NA values with median value of a column

df.bedrooms.median()

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
df


# In[27]:


# Create linear regression object.
reg = linear_model.LinearRegression()


# Fit the data.in other words , training the model based on the available model.
#  1st argument is has to be a 2D array. 
reg.fit(df.drop('price',axis='columns'),df.price)


# In[30]:


# Find price of home with 3000 sqr ft area, 3 bedrooms, 40 year old
reg.predict([[3000, 3, 40]])


# In[31]:


# Find price of home with 2500 sqr ft area, 4 bedrooms, 5 year old

reg.predict([[2500, 4, 5]])


# In[28]:


# coefficient of the linear regression line
reg.coef_


# In[32]:


# intercept of the linear regression line
reg.intercept_

