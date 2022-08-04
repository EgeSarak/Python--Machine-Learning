import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler=pd.read_csv("veriler.csv")


X=veriler[["boy","kilo","yas"]] #veya veriler.iloc[:,1:4] --->bagımsız degiskenler
y=veriler[["cinsiyet"]] #veya veriler.iloc[:,4:]-->bagımlı degiskenler

#train-test bölünmesi
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)

#öznitelik(değişken/kolon) ölçekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)

"""logistic regresyon"""
from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(X_train_scaled,y_train) #x ten y yi öğreniyoruz

y_pred=logr.predict(X_test_scaled)
print(y_pred)
print(y_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

"""KNN"""
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski")
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)

"""support vecktor classifier(SVC)"""
from sklearn.svm import SVC
svc=SVC(kernel="linear")
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print("SVC")
print(cm)

from sklearn.svm import SVC
svc=SVC(kernel="rbf")
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print("SVC")
print(cm)

from sklearn.svm import SVC
svc=SVC(kernel="poly")
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print("SVC")
print(cm)

"""Naif Bayes"""
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)

"""karar agacı(decisiontree) ile sınıflandırma"""
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)

"""Random Forest"""
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion="entropy")
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)


"""Roc egrisi"""
y_proba=rfc.predict_proba(X_test) #true ve false olma olasılıları
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr,tpr,thold=metrics.roc_curve(y_test,y_proba[:,0],pos_label="e")
print("fpr:",fpr)
print("tpr:",tpr)






















