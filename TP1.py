# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:31:42 2024

@author: Vasily
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# ####### Functions #######
# =============================================================================

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def decision_boundary(F_data, T_data, clf, title='', ylabel='', xlabel=''):
    fig, ax = plt.subplots()
    title = (title)
    X0, X1 = F_data[:, 0], F_data[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=T_data, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    plt.show()
    
def get_all_metrics(X_test, y_test, y_pred, model, F_data, T_data):
    cmatrix = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:\n', cmatrix)
    plt.figure()
    sns.heatmap(cmatrix, annot=True, cmap='RdYlBu')
    plt.title('Matrice de Confusion')

    proba = model.predict_proba(X_test)
    FPR, TPR, THS = roc_curve(y_test, proba[:,1])
    score = roc_auc_score(y_test, proba[:,1])
    print(f'Roc Auc Score = {score}')

    plt.figure()
    plt.plot(FPR, TPR, label=f'AUC = {score:.2}')
    plt.plot([x/100 for x in range(0,100)], [x/100 for x in range(0,100)],
             '--', color='r')
    plt.legend()
    plt.title("Courbe ROC")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    acc_score = accuracy_score(y_test, y_pred)
    print(f'Accuracy score = {acc_score}')
    
    decision_boundary(F_data, T_data, model, title='Frontière de décision')


# =============================================================================
# ####### Load Data #######
# =============================================================================
data = np.load('Data/ClassLearn1.npz')
lst = data.files
F_data = data['F']
T_data = data['T']

data2 = np.load('Data/ClassLearn2.npz')
lst2 = data2.files
F_data2 = data2['F']
T_data2 = data2['T']


# =============================================================================
# ####### Preliminary work #######
# =============================================================================

print('F data shape = ', F_data.shape)
print('T data shape = ', T_data.shape)

F_data_df = pd.DataFrame(data['F'])

pd.plotting.scatter_matrix(F_data_df)
plt.matshow(F_data_df.corr())
plt.show()

plt.figure()
sns.heatmap(F_data_df.corr(), annot=True, cmap='RdYlBu')


# =============================================================================
# ####### KNN #######
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(F_data, T_data, 
                                                    test_size=0.2)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

get_all_metrics(X_test, y_test, y_pred, model, F_data, T_data)

k = [1,3,5,7,11,15,21]
for i in k:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    print(f'Accuracy score for {i} neighbors = {acc_score}')


# =============================================================================
# ####### Naive Bayes #######
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(F_data, T_data, 
                                                      test_size=0.2)
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

get_all_metrics(X_test, y_test, y_pred, clf, F_data, T_data)


# =============================================================================
# ####### SVM #######
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(F_data, T_data, 
                                                    test_size=0.2)
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

get_all_metrics(X_test, y_test, y_pred, clf, F_data, T_data)

#########################################

X_train, X_test, y_train, y_test = train_test_split(F_data2, T_data2, 
                                                    test_size=0.2)
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

get_all_metrics(X_test, y_test, y_pred, clf, F_data2, T_data2)

#########################################

F_polaire = np.zeros((150,2))
F_polaire[:,0] = np.sqrt(F_data2[:,0]**2 + F_data2[:,1]**2)
F_polaire[:,1] = np.arctan(F_data2[:,1] / F_data2[:,0])


X_train, X_test, y_train, y_test = train_test_split(F_polaire, T_data2, 
                                                    test_size=0.2)
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

get_all_metrics(X_test, y_test, y_pred, clf, F_polaire, T_data)

#########################################

X_train, X_test, y_train, y_test = train_test_split(F_data2, T_data2, 
                                                    test_size=0.2)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

get_all_metrics(X_test, y_test, y_pred, clf, F_data2, T_data2)

#########################################

X_train, X_test, y_train, y_test = train_test_split(F_data2, T_data2, 
                                                    test_size=0.2)
for k in [3, 6, 10,15]:
    clf = svm.SVC(kernel='poly', degree=k, probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    get_all_metrics(X_test, y_test, y_pred, clf, F_data2, T_data2)


# =============================================================================
# ####### DecisionTreeClassifier #######
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(F_data, T_data, 
                                                    test_size=0.2)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plt.figure()
plot_tree(clf, fontsize=10)

get_all_metrics(X_test, y_test, y_pred, clf, F_data, T_data)


X_train, X_test, y_train, y_test = train_test_split(F_data2, T_data2, 
                                                    test_size=0.2)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plt.figure()
plot_tree(clf, fontsize=10)

get_all_metrics(X_test, y_test, y_pred, clf, F_data2, T_data2)



F_polaire = np.zeros((150,2))
F_polaire[:,0] = np.sqrt(F_data2[:,0]**2 + F_data2[:,1]**2)
F_polaire[:,1] = np.arctan(F_data2[:,1] / F_data2[:,0])


X_train, X_test, y_train, y_test = train_test_split(F_polaire, T_data2, 
                                                    test_size=0.2)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

get_all_metrics(X_test, y_test, y_pred, clf, F_polaire, T_data2)


RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(F_data, T_data, 
                                                    test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

get_all_metrics(X_test, y_test, y_pred, clf, F_data, T_data)


# =============================================================================
# ####### fetch_openml #######
# =============================================================================

data_all = fetch_openml(data_id=679, as_frame=True)
data = data_all['data']
target = data_all['target']
f_names = data_all['feature_names']

data_df = pd.DataFrame(data)
pd.plotting.scatter_matrix(data_df)

plt.figure()
sns.heatmap(data_df.corr(), annot=True, cmap='RdYlBu')

X_train, X_test, y_train, y_test = train_test_split(F_data, T_data, 
                                                    test_size=0.2)

# KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

get_all_metrics(X_test, y_test, y_pred, model, F_data, T_data)


# Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

get_all_metrics(X_test, y_test, y_pred, clf, F_data, T_data)


# SVM
clf = svm.SVC(kernel='rbf', degree=5, probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

get_all_metrics(X_test, y_test, y_pred, clf, F_data, T_data)


# DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plt.figure()
plot_tree(clf, fontsize=10)

get_all_metrics(X_test, y_test, y_pred, clf, F_data, T_data)


# RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

get_all_metrics(X_test, y_test, y_pred, clf, F_data, T_data)
