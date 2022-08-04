import numpy as np
import pandas as pd

yorumlar=pd.read_csv("Restaurant_Reviews.csv")

"""space matrix ve imla işaretleri"""
import re
yorum=re.sub(yorumlar["Review"][0])

"""büyük/küçük harf problemleri"""

yorum=yorum.lower()
yorum=yorum.split()

"""stop words"""
import nltk

durma=nltk.download("stopwords")

"""stemmer"""
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from nltk.corpus import stopwords
yorum=[ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]

yorum=" ".join(yorum)

#Preprocessin(Önişleme)
for i in range(1000):
    yorum=re.sub("[^a-zA-Z]"," ",yorumlar[Review][i])
    yorum.split()
    yorum=[ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum=" ".join(yorum)
    derlem=append(yorum)
    
"""CountVectorizer"""
#Feature Extraction(Öznitelik çıkarımı)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)
X=cv.fit_transform(derlem).toarray()
y=yorumlar.iloc[:,1]

"""makine öğrenmesi ve sınıflandırma kısmı"""
#makine öğrenmesi
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
model=gnb.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

     