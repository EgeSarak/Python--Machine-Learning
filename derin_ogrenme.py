import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Churn_Modelling.csv")
print(df)

#veri on isleme

X= df.iloc[:,3:13].values #bagımsız degiskenler
y = df.iloc[:,13].values #bagımlı degisken

#encoder: Kategorik -> Numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test) 

#yapay sinir agı
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(6,kernel_initializer="uniform",activation="relu",input_dim=11))

classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['accuracy'] )

classifier.fit(X_train_scaled, y_train, epochs=50)

y_pred = classifier.predict(X_test_scaled)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print(cm)