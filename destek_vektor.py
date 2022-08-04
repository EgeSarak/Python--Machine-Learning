#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

#♠öznitelik ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc2=sc2.fit_transform(Y)

#Svr
from sklearn.svm import SVR
svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)
#gorsellestirme
plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli))
#tahmin
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))
