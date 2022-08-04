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


"""linear regression"""
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()


"""polynomial regression"""
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

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

#öznitelik ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc2=sc2.fit_transform(Y)

"""Svr"""
from sklearn.svm import SVR
svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)
#gorsellestirme
plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli))
plt.show()
#tahmin
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

"""karar agacı"""
from sklearn.tree import DecisionTreeClassifier
r_dt=DecisionTreeClassifier(random_state=0)
r_dt.fit(X,Y)
Z=X+0.5
K=X-0.4
plt.scatter(X, Y, color="red")
plt.plot(x, r_dt.predict(X),color="black")

plt.plot(x,r_dt.predict(Z),color="green")
plt.plot(x,r_dt.predict(K),color="yellow")
plt.show()
#tahmin
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

"""Rassal agaclar(rf)"""
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())
#tahmin
print(rf_reg.predict([[6.6]]))
#gorsellestirme
plt.scatter(X, rf_reg.predict(X),color="red")
plt.plot(X,rf_reg.predict(X),color="blue")

plt.plot(X,rf_reg.predict(Z),color="green")
plt.plot(x, rf_reg.predict(K),color="yellow")
plt.show()



