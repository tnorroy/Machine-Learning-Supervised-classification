# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:28:31 2024

@author: Tom
"""

###importations

#%pip install --upgrade scikit-learn
#%pip install --upgrade imbalanced-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

###Fonctions

def missing_values(df):
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_table = pd.DataFrame({'Missing Values': missing_values, 
                                  'Percentage': missing_percent})
    print(missing_table)

def test_equilibre(y_train):
    occurrences_by_class = y_train.groupby(y_train).size()
    print(occurrences_by_class)
    
def one_hot_encoding(df, col):
     dff = pd.get_dummies(df, columns = col)
     return(dff)
    
def imputation(df, variable, imputed_variable):
    imputer_mean = SimpleImputer(strategy='mean')
    df[imputed_variable] = imputer_mean.fit_transform(df[[variable]])
    df.isnull().mean()
    
def compare_precision(X, y):
    logreg = LogisticRegression()
    logreg.fit(X, y)
    y_pred1 = logreg.predict(X)

    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X, y)
    y_pred2 = knn.predict(X)
    print("score pour knn avec k=5", metrics.accuracy_score(y, y_pred2), " vs score pour regression", 
      metrics.accuracy_score(y, y_pred1))

def decoupage_70_30(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    return(X_train, X_test, y_train, y_test)

def sur_echentillonage(X_train, X_test, y_train, y_test):
    rOs = RandomOverSampler()
    X_ro, y_ro = rOs.fit_resample(X_train, y_train)
    return(X_ro, y_ro)

def apprentissage_prediction_comparaison(X_train, X_test, y_train, y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print("Comparaison entre y_test et y_pred : ", metrics.accuracy_score(y_test, y_pred))
    return(y_pred)
  
def cross_validation(X_train, X_test, y_train, y_test):
    logreg = LogisticRegression()
    logreg.fit(X, y)
    score = cross_val_score(logreg, X_train, y_train, cv=8, scoring='accuracy')
    plt.plot(score)
    plt.show()
    
def confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.imshow(cm)

def qualite(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    
    

###Programme principal
df = pd.read_csv("C:/Users/Tom/Desktop/M1/Programmation/cirrhosis.csv", sep = ';')
print(df)

#Creation de y : test donnees equilibrees ou non et traitement  
y = df['Status']
test_equilibre(y)
y = y.replace('CL', 'C')
test_equilibre(y)


#Gestion des valeurs manquantes
missing_values(df)
imputation(df, 'Cholesterol', 'imp_Cholesterol')
imputation(df, 'Tryglicerides', 'imp_Tryglicerides')
imputation(df, 'Platelets', 'imp_Platelets')
imputation(df, 'Copper', 'imp_Copper')

#Creation du dataframe X
X_drop = df.drop(['Status', 'Cholesterol', 'Tryglicerides', 'Platelets', 'Copper'], axis = 1)
X = one_hot_encoding(X_drop, ['Stage','Drug', 'Sex', 'Ascites', 
                                   'Hepatomegaly', 'Spiders', 'Edema'])

#Decoupage des donnees puis sur-echantillonnage
X_train, X_test, y_train, y_test = decoupage_70_30(X, y)
X_train, y_train = sur_echentillonage(X_train, X_test, y_train, y_test)
test_equilibre(y_train)

#Recherche de la meilleur méthode
compare_precision(X, y)

#Apprentissage, création de y_pred et comparaison avec y_test
y_pred = apprentissage_prediction_comparaison(X_train, X_test, y_train, y_test)

#Test de la validite du modele
cross_validation(X_train, X_test, y_train, y_test)
confusion(y_test, y_pred)
qualite(y_test, y_pred)
