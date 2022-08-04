{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Aylar\n",
      "0       8\n",
      "1      10\n",
      "2      11\n",
      "3      13\n",
      "4      14\n",
      "5      19\n",
      "6      19\n",
      "7      20\n",
      "8      20\n",
      "9      24\n",
      "10     25\n",
      "11     25\n",
      "12     25\n",
      "13     26\n",
      "14     29\n",
      "15     31\n",
      "16     32\n",
      "17     34\n",
      "18     37\n",
      "19     37\n",
      "20     42\n",
      "21     44\n",
      "22     49\n",
      "23     50\n",
      "24     54\n",
      "25     55\n",
      "26     59\n",
      "27     59\n",
      "28     64\n",
      "29     65\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "df=pd.read_csv(\"satislar.csv\")\n",
    "\n",
    "X=df[[\"Aylar\"]] #bagımsız degisken\n",
    "print(X)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Satislar\n",
      "0    19671.5\n",
      "1    23102.5\n",
      "2    18865.5\n",
      "3    21762.5\n",
      "4    19945.5\n",
      "5    28321.0\n",
      "6    30075.0\n",
      "7    27222.5\n",
      "8    32222.5\n",
      "9    28594.5\n",
      "10   31609.0\n",
      "11   27897.0\n",
      "12   28478.5\n",
      "13   28540.5\n",
      "14   30555.5\n",
      "15   33969.0\n",
      "16   33014.5\n",
      "17   41544.0\n",
      "18   40681.5\n",
      "19   46970.0\n",
      "20   45869.0\n",
      "21   49136.5\n",
      "22   50651.0\n",
      "23   56906.0\n",
      "24   54715.5\n",
      "25   52791.0\n",
      "26   58484.5\n",
      "27   56317.5\n",
      "28   61195.5\n",
      "29   60936.0\n"
     ]
    }
   ],
   "source": [
    "y=df[[\"Satislar\"]] #bagımlı degisken\n",
    "print(y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Aylar\n",
      "5      19\n",
      "16     32\n",
      "8      20\n",
      "14     29\n",
      "23     50\n",
      "20     42\n",
      "1      10\n",
      "29     65\n",
      "6      19\n",
      "4      14\n",
      "18     37\n",
      "19     37\n",
      "9      24\n",
      "7      20\n",
      "25     55\n",
      "3      13\n",
      "0       8\n",
      "21     44\n",
      "15     31\n",
      "12     25     Aylar\n",
      "2      11\n",
      "28     64\n",
      "13     26\n",
      "10     25\n",
      "26     59\n",
      "24     54\n",
      "27     59\n",
      "11     25\n",
      "17     34\n",
      "22     49     Satislar\n",
      "5    28321.0\n",
      "16   33014.5\n",
      "8    32222.5\n",
      "14   30555.5\n",
      "23   56906.0\n",
      "20   45869.0\n",
      "1    23102.5\n",
      "29   60936.0\n",
      "6    30075.0\n",
      "4    19945.5\n",
      "18   40681.5\n",
      "19   46970.0\n",
      "9    28594.5\n",
      "7    27222.5\n",
      "25   52791.0\n",
      "3    21762.5\n",
      "0    19671.5\n",
      "21   49136.5\n",
      "15   33969.0\n",
      "12   28478.5     Satislar\n",
      "2    18865.5\n",
      "28   61195.5\n",
      "13   28540.5\n",
      "10   31609.0\n",
      "26   58484.5\n",
      "24   54715.5\n",
      "27   56317.5\n",
      "11   27897.0\n",
      "17   41544.0\n",
      "22   50651.0\n"
     ]
    }
   ],
   "source": [
    "#train-test bölünmesi\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)\n",
    "print(X_train,X_test,y_train,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-0.70368853]\n",
      " [ 0.15126015]\n",
      " [-0.63792324]\n",
      " [-0.0460357 ]\n",
      " [ 1.33503524]\n",
      " [ 0.80891298]\n",
      " [-1.29557607]\n",
      " [ 2.32151449]\n",
      " [-0.70368853]\n",
      " [-1.03251494]\n",
      " [ 0.48008657]\n",
      " [ 0.48008657]\n",
      " [-0.37486211]\n",
      " [-0.63792324]\n",
      " [ 1.66386166]\n",
      " [-1.09828023]\n",
      " [-1.42710664]\n",
      " [ 0.94044355]\n",
      " [ 0.08549487]\n",
      " [-0.30909683]]\n",
      "[[-1.68268756]\n",
      " [ 1.33023274]\n",
      " [-0.82997427]\n",
      " [-0.88682182]\n",
      " [ 1.04599497]\n",
      " [ 0.76175721]\n",
      " [ 1.04599497]\n",
      " [-0.88682182]\n",
      " [-0.37519385]\n",
      " [ 0.47751944]]\n",
      "[[-0.58893482]\n",
      " [-0.20450235]\n",
      " [-0.26937302]\n",
      " [-0.40591269]\n",
      " [ 1.75238875]\n",
      " [ 0.84837657]\n",
      " [-1.01636869]\n",
      " [ 2.08247565]\n",
      " [-0.44526921]\n",
      " [-1.27495041]\n",
      " [ 0.42348183]\n",
      " [ 0.93855664]\n",
      " [-0.56653314]\n",
      " [-0.67891012]\n",
      " [ 1.41533972]\n",
      " [-1.12612463]\n",
      " [-1.29739304]\n",
      " [ 1.11600906]\n",
      " [-0.12632172]\n",
      " [-0.5760344 ]]\n",
      "[[-1.66597621]\n",
      " [ 1.25819492]\n",
      " [-0.99762385]\n",
      " [-0.7856508 ]\n",
      " [ 1.0709181 ]\n",
      " [ 0.81055426]\n",
      " [ 0.92122098]\n",
      " [-1.04207705]\n",
      " [-0.09933754]\n",
      " [ 0.52977719]]\n"
     ]
    }
   ],
   "source": [
    "\"\"\"\n",
    "#öznitelik(değişken/kolon) ölçekleme\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "sc=StandardScaler()\n",
    "\n",
    "X_train=sc.fit_transform(x_train)\n",
    "print(X_train)\n",
    "X_test=sc.fit_transform(x_test)\n",
    "print(X_test)\n",
    "Y_train=sc.fit_transform(y_train)\n",
    "print(Y_train)\n",
    "Y_test=sc.fit_transform(y_test)\n",
    "print(Y_test)\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LinearRegression()\n"
     ]
    }
   ],
   "source": [
    "#model inşası\n",
    "from sklearn.linear_model import LinearRegression\n",
    "lr=LinearRegression()\n",
    "model=lr.fit(X_train,y_train)\n",
    "print(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[20991.93663769],\n",
       "       [62142.93172873],\n",
       "       [32638.44468232],\n",
       "       [31862.01081268],\n",
       "       [58260.76238052],\n",
       "       [54378.59303231],\n",
       "       [58260.76238052],\n",
       "       [31862.01081268],\n",
       "       [38849.91563946],\n",
       "       [50496.4236841 ]])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#uygulaması\n",
    "tahmin=lr.predict(X_test) #X_test değerlerini kullanrak y tahmin değerlerini bulduk\n",
    "tahmin"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'satışlar')"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#görselleştirme\n",
    "X_train=X_train.sort_index()\n",
    "y_train=y_train.sort_index()\n",
    "plt.plot(X_train,y_train)\n",
    "plt.plot(X_test,tahmin)\n",
    "\n",
    "plt.title(\"aylara göre satış\")\n",
    "plt.xlabel(\"aylar\")\n",
    "plt.ylabel(\"satışlar\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "8738b6b2e862f7e9341ea891b34e9c83c283a64354675045f4a9450407c0ce35"
  },
  "kernelspec": {
   "display_name": "Python 3.8.8 ('base')",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
