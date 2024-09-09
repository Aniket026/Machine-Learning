# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:23:38 2024

@author: aniket
"""


 
from sklearn.cluster import KMeans 
import pandas  as pd 
import numpy as np
import matplotlib.pylab as plt

#let us try to undersatnd first hoe k means works for two diamentinal data 
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
#create a empty dataframe with o rpw and 0 columns
df_xy=pd.DataFrame(columns=["X","Y"])
#assign the value of X and  Y to these columns


df_xy.X=X
df_xy.Y=Y


df_xy.plot(x="X",y="Y",kind="scatter")
model1=KMeans(n_clusters=3).fit(df_xy)

'''
with data x and y apply kmeans model,
generate scatter plot
camap=cm.coolwarm:cool color combination
'''
model1.labels_
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)

####################################################################################






Univ1=pd.read_excel("C:/7-Clustering/University_Clustering.xlsx")
Univ1.describe()
Univ1.head()
Univ=Univ1.drop(["State"],axis=1)



def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_func(Univ.iloc[:,1:])

TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    
    TWSS.append(kmeans.inertia_)

TWSS

#as k value increases the TWSS value decreases

plt.plot(k,TWSS,'ro-');
plt.xlabel("No_of_clusters")
plt.ylabel("Total_within_SS")

'''
How to select value of k from elbow curve when k changes from 2 and 3
,then decrease in twss in higher than when k changes from 3 to 4
when k values changes from 5 to 6 decrease
''' 




model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
Univ['clust']=mb
Univ.head()
Univ=Univ.iloc[:,[7,0,1,2,3,4,5,6,]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()
Univ.to_csv("kmeans_university.csv",encoding='utf-8')
import os
os.getcwd()













