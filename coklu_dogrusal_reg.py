import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("veriler.csv")

#eksik verileri ortalama ile doldurduk
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

yas=df.iloc[:,1:4].values #boy kilo yaş kolonlarını aldık
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)

#kategorikten numerice dönüştürme işlemi
from sklearn import preprocessing
ulke=df.iloc[:,0:1].values
print(ulke)

le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(df.iloc[:,0])
print(ulke[:,0])

one_hot_encoding=preprocessing.OneHotEncoder()
ulke=one_hot_encoding.fit_transform(ulke).toarray()
print(ulke)

#kategorikten numerice dönüştürme işlemi
from sklearn import preprocessing
c=df.iloc[:,-1:].values
print(c)

le=preprocessing.LabelEncoder()
c[:,-1]=le.fit_transform(df.iloc[:,-1])
print(c)


one_hot_encoding=preprocessing.OneHotEncoder()
c=one_hot_encoding.fit_transform(c).toarray()
print(c)


#numpy dizilerini dataframe dönüştürme
print(list(range(22)))
sonuc=pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
print(sonuc)

sonuc2=pd.DataFrame(data=yas,index=range(22),columns=["boy","kilo","yaş"])
print(sonuc2)

cinsiyet=df.iloc[:,-1].values #son sütunu aldık
print(cinsiyet)

sonuc3=pd.DataFrame(data=c[:,:1],index=range(22),columns=["Cinsiyet"])
print(sonuc3)

#birleşitrme
s=pd.concat([sonuc,sonuc2],axis=1) #kolon olarak ekledik
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#train-test bölünmesi
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)


#model oluşturma
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

#ayrıma işlemi yaptık
boy=s2.iloc[:,3:4].values #veya  y=df[["boy"]] #bagımlı degisken
sol=s2.iloc[:,:3] #bagımsız
sag=s2.iloc[:,4:] #bagımsız 

#bağımlı değişkeni çıkardıktan sonra bu işlemi yaparak bagımsız degiskenleri birleştirdik
veri=pd.concat([sol,sag],axis=1)

X_train,X_test,y_train,y_test=train_test_split(veri,boy,test_size=0.33,random_state=0)


reg2=LinearRegression()
reg2.fit(X_train,y_train) #x_trainden y_traini öğren
y_pred=reg2.predict(X_test) #X_test bağımsız değerleri kullanarak y tahmin değerlerini elde ettik


"""Python ile Geri Eleme(Backward Elimination)"""

import statsmodels.api as sm

X=np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)
X_list=veri.iloc[:,[0,1,2,3,4,5]].values
X_list=np.array(X_list,dtype=float)
model=sm.OLS(boy,X_list).fit()
print(model.summary())

#x5 in p valuesi cok yuksek cıktıgı için atıcaz

X=np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)
X_list=veri.iloc[:,[0,1,2,3,5]].values
X_list=np.array(X_list,dtype=float)
model=sm.OLS(boy,X_list).fit()
print(model.summary())

 












