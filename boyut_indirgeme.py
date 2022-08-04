"""PCA"""
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri kümesi
veriler = pd.read_csv('Wine.csv')
X=veriler.iloc[:,:13].values #bagımsız degiskenler
y=veriler.iloc[:,13].values #bagımlı degisken

# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

#pca dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_scaled,y_train)

#pca dönüşümünden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train_pca,y_train)

#tahminler
y_pred = classifier.predict(X_test_scaled)

y_pred2 = classifier2.predict(X_test_pca)

from sklearn.metrics import confusion_matrix
#actual / PCA olmadan çıkan sonuç
print('gercek / PCAsiz')
cm = confusion_matrix(y_test,y_pred)
print(cm)

#actual / PCA sonrası çıkan sonuç
print("gercek / pca ile")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

#PCA sonrası / PCA öncesi
print('pcasiz ve pcali')
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)

"""LDA"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
X_train_lda=lda.fit_transform(X_train_scaled,y_train)
X_test_lda=lda.transform(X_test_scaled)

#LDA donusumunden sonra
classifier_lda=LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)
#LDA verisini tahmin et
y_pred_lda=classifier_lda.predict(X_test_lda)

#LDA sonrası/orjinal
print('lda ve orjinal')
cm4 = confusion_matrix(y_pred,y_pred_lda)
print(cm4)





















