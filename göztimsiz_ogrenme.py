import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("musteriler.csv")

X=veriler.iloc[:,3:].values #veya veriler[["Hacim","maas"]]


"""KMeans"""
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3,init="k-means++")
kmeans.fit(X)
print(kmeans.cluster_centers_)

"""
#wcss kullanarak k için optimum deger bulma
sonuclar=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)    
"""



#KMeans
kmeans=KMeans(n_clusters=4,init="k-means++",random_state=123)
y_pred=kmeans.fit(X)
print(y_pred)
plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c="red")
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c="blue")
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c="green")
plt.title("Kmeans")
plt.show()


"""Hierarchical Clustering"""
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
y_pred=ac.fit_predict(X)
print(y_pred)

plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c="red")
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c="blue")
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c="green")
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100,c="yellow")
plt.title("HC")
plt.show()

#dendrogram
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method="ward"))
plt.show()