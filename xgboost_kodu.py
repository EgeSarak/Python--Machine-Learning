#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri kümesi
df=pd.read_csv("Churn_Modelling.csv")
X=df.iloc[:,3:13].values
y=df.iloc[:,13].values

#önişleme
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#xgboost
#!pip install xgboost
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)