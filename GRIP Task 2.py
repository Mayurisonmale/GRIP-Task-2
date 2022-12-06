#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation: Data Science and Business Analytics Intership
# ## Prediction using Unsupervised ML Problem
# ### From the given iris dataset predict the optimum number of clusters and represent it visually 
# ### Author: Mayuri Umesh Sonmale

# # Step 1: Importing The Dataset

# In[1]:


# importing required libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[2]:


# Load the Dataset
data=pd.read_csv("C:/Users/HP-PC/Desktop/Mayuri/IRIS.csv")
data.head()#To show the first five coulmns of data set


# In[3]:


data.drop("Id",axis=1,inplace=True)


# # Step 2:Data Wrangling

# In[4]:


data.describe()


# In[5]:


data.info()


# In[7]:


data.Species.value_counts()


# # Step 3:Using the Elow Method to find the optimal number of clusters

# In[11]:


# find the optimal number of clusters for k means clustering
x=data.iloc[:,:-1].values
from sklearn.cluster import KMeans
WCSS=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',
                 max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11),WCSS)
plt.title('The Elow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ###  We choose the number of cluster as '3'

# # Step 4: Training the kmeans Model on the dataset

# In[12]:


# Apply kmeans to the dataset
kmeans=KMeans(n_clusters=3,
             max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)
y_kmeans


# # Step 5: Visualize the test set result

# In[14]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],
           s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],
           s=100,c='blue',label='Iris-versicolour')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],
           s=100,c='green',label='Iris-viginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
           s=100,c='yellow',label='Centroids')
plt.legend()


# # Data Visualisation

# In[15]:


data.corr()


# In[16]:


plt.figure(figsize=(12,5))
sns.heatmap(data.corr(),annot=True,cmap="BuPu")


# In[18]:


plt.figure(figsize=(8,6))
sns.boxplot(x="Species",y="SepalLengthCm",data=data)


# In[19]:


sns.pairplot(data.corr())


# # Thank You!

# In[ ]:




