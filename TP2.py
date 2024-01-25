# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:25:07 2024

@author: Vasily
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import warnings
warnings.filterwarnings("ignore")


data = np.load('Data_TP2/Rlearn1.npz')
lst = data.files

print('----- Rlearn1 data -----')
Vl = data['Vl']
Il = data['Il']
Rref = data['Rref']

plt.figure(figsize=(14,10))
plt.scatter(Il, Vl, label='Measurement model')
plt.scatter(Il, Il*Rref, label='Theoretical model')
plt.title('Measurement & Theoretical models for Rlearn1 data')
plt.xlabel('Current')
plt.ylabel('Voltage')
plt.legend()
plt.show()


M = LinearRegression()
Il = Il.reshape(-1, 1)
M.fit(Il, Vl)
print(f'Estimated R value = {M.coef_[0]}')

Vl_pred = M.predict(Il)

plt.figure(figsize=(14,10))
plt.scatter(Il, Vl_pred, label='Predicted model')
plt.scatter(Il, Il*Rref, label='Theoretical model')
plt.title('Predicted & Theoretical models for Rlearn1 data')
plt.xlabel('Current')
plt.ylabel('Voltage')
plt.legend()
plt.show()

mse = mean_squared_error(Vl, Vl_pred)
print(f'MSE = {mse}')
r2 = r2_score(Vl, Vl_pred)
print(f'r2_score = {r2}\n')


print('----- Rlearn2 data -----')

data2 = np.load('Data_TP2/Rlearn2.npz')
lst2 = data2.files

Vl2 = data2['Vl']
Il2 = data2['Il']
Rref2 = data2['Rref']

plt.figure(figsize=(14,10))
plt.scatter(Il2, Vl2, label='Measurement model')
plt.scatter(Il2, Il2*Rref2, label='Theoretical model')
plt.title('Measurement & Theoretical models for Rlearn2 data')
plt.xlabel('Current')
plt.ylabel('Voltage')
plt.legend()
plt.show()


M2 = LinearRegression()
Il2 = Il2.reshape(-1, 1)
M2.fit(Il2, Vl2)
print(f'Estimated R value = {M2.coef_[0]}')

Vl_pred2 = M2.predict(Il2)

plt.figure(figsize=(14,10))
plt.scatter(Il2, Vl_pred2, label='Predicted model')
plt.scatter(Il2, Il2*Rref2, label='Theoretical model')
plt.title('Predicted & Theoretical models for Rlearn2 data')
plt.xlabel('Current')
plt.ylabel('Voltage')
plt.legend()
plt.show()

mse2 = mean_squared_error(Vl2, Vl_pred2)
print(f'MSE = {mse2}')
r2_2 = r2_score(Vl2, Vl_pred2)
print(f'r2_score = {r2_2}\n')


print('----- Difference between two datasets -----')
print(f'R difference = {abs(M2.coef_[0] - M.coef_[0])}')
print(f'MSE difference = {abs(mse2 - mse)}')
print(f'R2 difference = {abs(r2_2 - r2)}\n')


print('----- 5th order polynomial model (Rlearn1) -----')

poly = PolynomialFeatures(5)
Ilp = poly.fit_transform(Il)
M3 = LinearRegression()
M3.fit(Ilp, Vl)
Vlp_pred_5 = M3.predict(Ilp)

mse5 = mean_squared_error(Vlp_pred_5, Vl)
print(f'MSE values : 5th = {mse5}, 1st = {mse}')
r2_5 = r2_score(Vlp_pred_5, Vl)
print(f'R2 values : 5th = {r2_5}, 1st = {r2}\n')


plt.figure(figsize=(14,10))
plt.scatter(Il, Il*Rref, label='Theoretical model')
plt.scatter(Il, Vl, label='Measurement model')
plt.scatter(Il, Vl_pred, label='Predicted model (order 1)')
plt.scatter(Il, Vlp_pred_5, label='Predicted model (order 5)')
plt.title('Models for Rlearn1 data')
plt.xlabel('Current')
plt.ylabel('Voltage')
plt.legend()
plt.show()


print('----- 5th order polynomial model (Rlearn2) -----')

poly = PolynomialFeatures(5)
Ilp2 = poly.fit_transform(Il2)
M4 = LinearRegression()
M4.fit(Ilp2, Vl2)
Vlp2_pred_5 = M4.predict(Ilp2)

mse2_5 = mean_squared_error(Vlp2_pred_5, Vl2)
print(f'MSE values : 5th = {mse2_5}, 1st = {mse2}')
r2_2_5 = r2_score(Vlp2_pred_5, Vl2)
print(f'R2 values : 5th = {r2_5}, 1st = {r2_2_5}')


plt.figure(figsize=(14,10))
plt.scatter(Il2, Il2*Rref, label='Theoretical model')
plt.scatter(Il2, Vl2, label='Measurement model')
plt.scatter(Il2, Vl_pred2, label='Predicted model (order 1)')
plt.scatter(Il2, Vlp2_pred_5, label='Predicted model (order 5)')
plt.title('Models for Rlearn2 data')
plt.xlabel('Current')
plt.ylabel('Voltage')
plt.legend()
plt.show()


# ---------- Regression for COVID19 case estimation ----------

data_covid = np.load('Data_TP2/CovidData.npz')
lst_covid = data_covid.files
Date = data_covid['Date']
Date_df = pd.DataFrame(data_covid['Date'])
Case = data_covid['Case']
Case_df = pd.DataFrame(data_covid['Case'])
Death = data_covid['Death']
Death_df = pd.DataFrame(data_covid['Death'])

day = pd.Timestamp('2020-07-21')
index_21_july_2020 = Date_df[Date_df[0] == day].index[0]
print(f'Index of 21 July 2020 = {index_21_july_2020}\n')

plt.figure(figsize=(14,10))
plt.plot(Death[index_21_july_2020:], Case[index_21_july_2020:], 'o')
plt.xlabel('Number of Death')
plt.ylabel('Number of Cases')
plt.show()

model = LinearRegression()
model.fit(Death[index_21_july_2020:].reshape(-1,1), Case[index_21_july_2020:])
Case_pred = model.predict(Death.reshape(-1,1))

plt.figure(figsize=(14,10))
plt.scatter(Death.reshape(-1,1), Case, label='Real Data')
plt.plot(Death.reshape(-1,1), Case_pred, label='Predicted Data', color='red')
plt.title('Number of cases vs number of deaths')
plt.xlabel('Number of Death')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()


cas_cumulate = np.cumsum(Case)
death_cumulate = np.cumsum(Death).reshape(-1,1)

plt.figure(figsize=(14,10))
plt.plot(death_cumulate[index_21_july_2020:], cas_cumulate[index_21_july_2020:], 'o')
plt.title('Cumulative number of cases vs deaths')
plt.xlabel('Number of Death')
plt.ylabel('Number of Cases')
plt.show()  


print('----- Polynomial model -----')

poly_3 = PolynomialFeatures(3)
death_poly = poly_3.fit_transform(death_cumulate)
model_poly = LinearRegression()
model_poly.fit(death_poly[index_21_july_2020:], cas_cumulate[index_21_july_2020:])
case_poly = model_poly.predict(death_poly)

plt.figure(figsize=(14,10))
plt.scatter(death_cumulate, cas_cumulate, label='Real Data')
plt.plot(death_cumulate, case_poly, label='Predicted Data (order 3)', color='red')
plt.title('Cumulative number of cases vs deaths')
plt.xlabel('Number of Death')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()


plt.figure(figsize=(14,10))
plt.scatter(death_cumulate[index_21_july_2020:], cas_cumulate[index_21_july_2020:], 
            label='Real Data')
plt.plot(death_cumulate[index_21_july_2020:], case_poly[index_21_july_2020:], 
          label='Predicted Data (order 3)', color='red')
plt.title('Cumulative number of cases vs deaths (zoom)')
plt.xlabel('Number of Death')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()



# ---------- Clustering ----------

data_clust = np.load('Data_TP2/Cdata1.npz')
lst_clust = data_clust.files

X = data_clust['X']
X_df = pd.DataFrame(X)
C = data_clust['C']

X0 = X[C == 0]
X1 = X[C == 1]
X2 = X[C == 2]

plt.figure(figsize=(14,10))
plt.scatter(X[:,0],X[:,1])
plt.xlabel('1st Feature')
plt.ylabel('2nd Feature')
plt.title('Cloud of points')
plt.show()

plt.figure(figsize=(14,10))
plt.scatter(X0[:,0],X0[:,1], label='Class 1')
plt.scatter(X1[:,0],X1[:,1], label='Class 2')
plt.scatter(X2[:,0],X2[:,1], label='Class 3')
plt.xlabel('1st Feature')
plt.ylabel('2nd Feature')
plt.title('Clusters')
plt.legend()
plt.show()

print(f'Mean value for 1st Class = {np.mean(X0, axis=0)}')
print(f'Mean value for 2st Class = {np.mean(X1, axis=0)}')
print(f'Mean value for 3st Class = {np.mean(X2, axis=0)}\n')

plt.figure()
sns.heatmap(np.cov(X0[:,0],X0[:,1]), annot=True, cmap='RdYlBu')
plt.title('Covariance matrice for 1st Class')
plt.figure()
sns.heatmap(np.cov(X1[:,0],X1[:,1]), annot=True, cmap='RdYlBu')
plt.title('Covariance matrice for 2nd Class')
plt.figure()
sns.heatmap(np.cov(X2[:,0],X2[:,1]), annot=True, cmap='RdYlBu')
plt.title('Covariance matrice for 3rd Class')


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
clusters = kmeans.labels_

X0_pred = X[clusters == 0]
X1_pred = X[clusters == 1]
X2_pred = X[clusters == 2]

plt.figure(figsize=(14,10))
plt.scatter(X0_pred[:,0],X0_pred[:,1], label='Class 1')
plt.scatter(X1_pred[:,0],X1_pred[:,1], label='Class 2')
plt.scatter(X2_pred[:,0],X2_pred[:,1], label='Class 3')
plt.xlabel('1st Feature')
plt.ylabel('2nd Feature')
plt.title('Predicted Clusters')
plt.legend()
plt.show()

print('Predicted Values')
print(f'Mean value for 1st Class = {np.mean(X0_pred, axis=0)}')
print(f'Mean value for 2st Class = {np.mean(X1_pred, axis=0)}')
print(f'Mean value for 3st Class = {np.mean(X2_pred, axis=0)}\n')


# Data 2
data_clust = np.load('Data_TP2/Cdata2.npz')
lst_clust = data_clust.files

X = data_clust['X']
X_df = pd.DataFrame(X)
C = data_clust['C']

X0 = X[C == 0]
X1 = X[C == 1]
X2 = X[C == 2]

plt.figure(figsize=(14,10))
plt.scatter(X[:,0],X[:,1])
plt.xlabel('1st Feature')
plt.ylabel('2nd Feature')
plt.title('Cloud of points')
plt.show()

plt.figure(figsize=(14,10))
plt.scatter(X0[:,0],X0[:,1], label='Class 1')
plt.scatter(X1[:,0],X1[:,1], label='Class 2')
plt.scatter(X2[:,0],X2[:,1], label='Class 3')
plt.xlabel('1st Feature')
plt.ylabel('2nd Feature')
plt.title('Clusters')
plt.legend()
plt.show()

print(f'Mean value for 1st Class = {np.mean(X0, axis=0)}')
print(f'Mean value for 2st Class = {np.mean(X1, axis=0)}')
print(f'Mean value for 3st Class = {np.mean(X2, axis=0)}\n')

plt.figure()
sns.heatmap(np.cov(X0[:,0],X0[:,1]), annot=True, cmap='RdYlBu')
plt.title('Covariance matrice for 1st Class')
plt.figure()
sns.heatmap(np.cov(X1[:,0],X1[:,1]), annot=True, cmap='RdYlBu')
plt.title('Covariance matrice for 2nd Class')
plt.figure()
sns.heatmap(np.cov(X2[:,0],X2[:,1]), annot=True, cmap='RdYlBu')
plt.title('Covariance matrice for 3rd Class')


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
clusters = kmeans.labels_

X0_pred = X[clusters == 0]
X1_pred = X[clusters == 1]
X2_pred = X[clusters == 2]

plt.figure(figsize=(14,10))
plt.scatter(X0_pred[:,0],X0_pred[:,1], label='Class 1')
plt.scatter(X1_pred[:,0],X1_pred[:,1], label='Class 2')
plt.scatter(X2_pred[:,0],X2_pred[:,1], label='Class 3')
plt.xlabel('1st Feature')
plt.ylabel('2nd Feature')
plt.title('Predicted Clusters')
plt.legend()
plt.show()

print('Predicted Values')
print(f'Mean value for 1st Class = {np.mean(X0_pred, axis=0)}')
print(f'Mean value for 2st Class = {np.mean(X1_pred, axis=0)}')
print(f'Mean value for 3st Class = {np.mean(X2_pred, axis=0)}')


# Data 3

data_clust = np.load('Data_TP2/Cdata3.npz')
lst_clust = data_clust.files

X = data_clust['X']
X_df = pd.DataFrame(X)
C = data_clust['C']

X0 = X[C == 0]
X1 = X[C == 1]
X2 = X[C == 2]


plt.figure(figsize=(14,10))
plt.scatter(X[:,0],X[:,1])
plt.xlabel('1st Feature')
plt.ylabel('2nd Feature')
plt.title('Cloud of points')
plt.show()

plt.figure(figsize=(14,10))
plt.scatter(X0[:,0],X0[:,1], label='Class 1')
plt.scatter(X1[:,0],X1[:,1], label='Class 2')
plt.scatter(X2[:,0],X2[:,1], label='Class 3')
plt.xlabel('1st Feature')
plt.ylabel('2nd Feature')
plt.title('Clusters')
plt.legend()
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
clusters = kmeans.labels_

X0_pred = X[clusters == 0]
X1_pred = X[clusters == 1]
X2_pred = X[clusters == 2]

plt.figure(figsize=(14,10))
plt.scatter(X0_pred[:,0],X0_pred[:,1], label='Class 1')
plt.scatter(X1_pred[:,0],X1_pred[:,1], label='Class 2')
plt.scatter(X2_pred[:,0],X2_pred[:,1], label='Class 3')
plt.xlabel('1st Feature')
plt.ylabel('2nd Feature')
plt.title('Predicted Clusters')
plt.legend()
plt.show()


gm = GaussianMixture(n_components=3)
gm.fit(X)
gm_clusters = gm.predict(X)

X0_pred_gm = X[gm_clusters == 0]
X1_pred_gm = X[gm_clusters == 1]
X2_pred_gm = X[gm_clusters == 2]

plt.figure(figsize=(14,10))
plt.scatter(X0_pred_gm[:,0],X0_pred_gm[:,1], label='Class 1')
plt.scatter(X1_pred_gm[:,0],X1_pred_gm[:,1], label='Class 2')
plt.scatter(X2_pred_gm[:,0],X2_pred_gm[:,1], label='Class 3')
plt.xlabel('1st Feature')
plt.ylabel('2nd Feature')
plt.title('Predicted Clusters with GaussianMixture')
plt.legend()
plt.show()


# ---------- Image segmentation ----------

data_image = np.load('Data_TP2/Image1.npz')
lst_clust = data_image.files
I = data_image['I']

plt.matshow(I)

I_vect = I.ravel()
plt.figure(figsize=(14, 10))
plt.hist(I_vect, bins=200)

I_vect = I_vect.reshape(-1, 1)
kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(I_vect)

label_image = labels.reshape(I.shape)
plt.matshow(label_image)
plt.show()


# Data 2

data_image = np.load('Data_TP2/Image2.npz')
lst_clust = data_image.files
I = data_image['I']

plt.matshow(I)

I_vect = I.ravel()
plt.figure(figsize=(14, 10))
plt.hist(I_vect, bins=200)

I_vect = I_vect.reshape(-1, 1)
kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(I_vect)

label_image = labels.reshape(I.shape)
plt.matshow(label_image)
plt.show()
