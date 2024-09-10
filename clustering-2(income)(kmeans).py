# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 08:57:54 2024

@author: aniket
"""



from sklearn.cluster import KMeans 
import pandas  as pd 
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt


df=pd.read_csv("C:/7-Clustering/income.csv")
df.head()

plt.scatter(df.Age,df["Income($)"])
plt.xlabel("Age")
plt.ylabel("Income($)")

km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[["Age","Income($)"]])
y_predicted
df["cluster"]=y_predicted
df.head()
km.cluster_centers_




df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')

plt.scatter(df2.Age,df2['Income($)'],color='red')

plt.scatter(df3.Age,df3['Income($)'],color='black')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()


#preprocessing using min max scaler

scaler=MinMaxScaler()

scaler.fit(df[["Income($)"]])
df["Income($)"]=scaler.transform(df[["Income($)"]])


scaler.fit(df[["Age"]])
df["Age"]=scaler.transform(df[["Age"]])

df.head()

plt.scatter(df.Age,df["Income($)"])

km=KMeans(n_clusters=3)

y_predicted=km.fit_predict(df[["Age","Income($)"]])
y_predicted
