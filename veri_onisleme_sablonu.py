import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#eksik verileri ortalama ile doldurduk
from sklearn.impute import SimpleImputer
veriler=pd.read_csv("eksikveriler.csv")
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

yas=veriler.iloc[:,1:4].values #boy kilo yaş kolonlarını aldık
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)

#kategorikten numerice dönüştürme işlemi
from sklearn import preprocessing
ulke=veriler.iloc[:,0:1].values
print(ulke)

le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke[:,0])

one_hot_encoding=preprocessing.OneHotEncoder()
ulke=one_hot_encoding.fit_transform(ulke).toarray()
print(ulke)

#numpy dizilerini dataframe dönüştürme
print(list(range(22)))
sonuc=pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
print(sonuc)

sonuc2=pd.DataFrame(data=yas,index=range(22),columns=["boy","kilo","yaş"])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values #son sütunu aldık
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=["Cinsiyet"])
print(sonuc3)

#birleşitrme
s=pd.concat([sonuc,sonuc2],axis=1) #kolon olarak ekledik
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#train-test bölünmesi
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)
print(X_train,X_test,y_train,y_test)

#öznitelik(değişken/kolon) ölçekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(X_train)
print(x_train)
x_test=sc.fit_transform(X_test)
print(x_test)

