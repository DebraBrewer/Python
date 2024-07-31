#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) #Adjusts the configurations of the plots

# Read in the data
df = pd.read_csv(r'C:\Users\debr8\Downloads\movies.csv')


# In[4]:


# Look at data
df.head()


# In[5]:


# determine what data is missing

for col in df.columns:
    percent_miss = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,percent_miss))


# In[6]:


# drop rows with missing data

df = df.dropna()


# In[7]:


# See labels and data types for columns
pd.set_option('display.max_rows', 20)
df.dtypes


# In[8]:


# Test if column is the same when rounded
df['budget'].equals(df['budget'].round())
df['gross'].equals(df['gross'].round())
df['runtime'].equals(df['runtime'].round())
df['votes'].equals(df['votes'].round())


# In[9]:


# Change from float to int
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')
df['runtime']=df['runtime'].astype('int64')
df['votes']=df['votes'].astype('int64')


# In[10]:


# Pull out correct year from "released" as it does not match the current "year" column
df['releasedyear'] = df['released'].astype(str).str.split().str[2]


# In[11]:


# Showing that the year obtained from "released" column doesn't match year from "year"
df['releasedyear'].equals(df['year'])


# In[40]:


pd.set_option('display.max_rows', 10)
df


# In[12]:


# drop any duplicates

df.drop_duplicates()


# In[13]:


# check for quality of data
pd.set_option('display.max_rows', None)
df['company'].drop_duplicates().sort_values(ascending=True)


# In[14]:


df.sort_values(by=['gross'], inplace=False, ascending=False)
df.head()


# In[27]:


# explore correlations


# In[25]:


# correlation matrix, method options = #pearson, kendall,spearman
# determine highest correlation between data
df.corr(numeric_only=True, method = 'pearson') 


# In[26]:


# heatmap of correlation matrix

correlation_matrix = df.corr(numeric_only=True, method = 'pearson') 
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[15]:


# scatterplot of budget vs gross revenue 
# correlation heatmap gives this as highest correlation (0.74)

plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')

plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')

# Plot budget vs gross using seaborn
sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color":"red"}, line_kws={"color":"blue"})

plt.show()

