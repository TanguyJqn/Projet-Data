# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:50:12 2019

@author: Tanguy Jennequin
"""

import numpy as np
import pandas as pd

def percep(data,limite = 4000,learning_rate = 0.01):
    
    train = entrainement(data)
    data_train = train.iloc[:,:11]
    data_train_quality = train.iloc[:,11:]
    
    evalue = test(data)
    data_test = evalue.iloc[:,:11]
    data_test_quality = evalue.iloc[:,11:]

    data_train_np = np.array(data_train)
    data_test_np = np.array(data_test)
    data_train_quality_np = np.array(data_train_quality)
    data_test_quality_np = np.array(data_test_quality)

    p = Perceptron(11,limite,learning_rate)
    p.train(data_train_np,data_train_quality_np)
    score = p.score_perceptron(data_test_np,data_test_quality_np)

    return score     


class Perceptron(object):

    def __init__(self, nb_inputs, seuil=4000, learning_rate=0.01):
        self.seuil = seuil
        self.learning_rate = learning_rate
        self.poids = np.zeros(nb_inputs + 1) #poids du biais

    def predict(self, inputs):
        somme = np.dot(self.poids[0:11],inputs.T) + self.poids[-1] #f(x) =1 si poids.inputs + biais > 0
                                              # il faut que inputs et weights aient les mêmes dimensions (produit scalaire)
        if somme > 0:  

            activation = 1   # vin de qualité, le label(résultat véridique) est de 1

        else:
            activation = -1  # vin de mauvaise qualité, le label (résultat véridique) est de -1

        return activation

    def train(self, training_inputs, labels):

        for lim in range(self.seuil): 
            """labels numpy tableau output vauluesfor toutes les valeurs correspondantes a training inputs"""
            for inputs, label in zip(training_inputs, labels): 
                """training inputs mm taille que labels // labels - > label et training input ->  input"""
                prediction = self.predict(inputs)
                self.poids[0:11] += self.learning_rate * (label - prediction) * (inputs)
                """label - prediction error"""
                self.poids[-1] += self.learning_rate * (label - prediction)
                """biais->"""
                
    def score_perceptron(self, test_data, resultat):

        i = 0.0             # initializing to float to avoid caclculus issues with euclidian division
        nb_error = 0.0
        i = test_data.shape[0]      # takes the number of row in the testing_data

        for row, res in zip(test_data, resultat):
            
            prediction = self.predict(row)

            if prediction != res:   # if the predicted class is different than the actual class,
                nb_error += 1  # increments nb_error
                
        return 1 - (nb_error/i)
    
    
    

def integ():
    data = pd.read_csv("red_wines.csv")
    #data = clean(data)
    return data

    
def mauvais(data):
    return data[data['quality']==-1]

def bon(data):
    return data[data['quality']==1]

   
def median(data_m,data_b):
    
    data_mauvais = data_m.median(skipna = True)
    data_bon = data_b.median(skipna = True)
    
    frames = [data_mauvais,data_bon]
    
    return pd.concat(frames,axis = 1, sort = False)
"""
def pH(data):    
    
     return data[(data['pH']<=3.5) & (data['pH']>=3)]
"""

def fixed_acidity(data):
    
    return data[(data['fixed acidity']>=data['fixed acidity'].mean() -3*data['fixed acidity'].std()) &
                (data['fixed acidity']<= data['fixed acidity'].mean()+ 3*data['fixed acidity'].std())]
    
def volatile_acidity(data):
    
    return data[(data['volatile acidity']>=data['volatile acidity'].mean() -3*data['volatile acidity'].std()) &
                (data['volatile acidity']<= data['volatile acidity'].mean()+ 3*data['volatile acidity'].std())]
    
def citric_acid(data):
    
    return data[(data['citric acid']>=data['citric acid'].median() -3*data['citric acid'].std()) &
                (data['citric acid']<= data['citric acid'].median()+ 3*data['citric acid'].std())]

def residual_sugar(data):
    
    return data[(data['residual sugar']>=data['residual sugar'].mean() -3*data['residual sugar'].std()) &
                (data['residual sugar']<= data['residual sugar'].mean()+ 3*data['residual sugar'].std())]
    
def chlorides(data):
    
    return data[(data['chlorides']>=data['chlorides'].mean() -3*data['chlorides'].std()) &
                (data['chlorides']<= data['chlorides'].mean()+ 3*data['chlorides'].std())]
    
def free_sulfur_dioxide(data):
    
    return data[(data['free sulfur dioxide']>=data['free sulfur dioxide'].median() -3*data['free sulfur dioxide'].std()) &
                (data['free sulfur dioxide']<= data['free sulfur dioxide'].median()+ 3*data['free sulfur dioxide'].std())]
    
def total_sulfur_dioxide(data):
    
    return data[(data['total sulfur dioxide']>=data['total sulfur dioxide'].median() -3*data['total sulfur dioxide'].std()) &
                (data['total sulfur dioxide']<= data['total sulfur dioxide'].median()+ 3*data['total sulfur dioxide'].std())]
    
def density(data):
    
    return data[(data['density']>=data['density'].mean() -3*data['density'].std()) &
                (data['density']<= data['density'].mean()+ 3*data['density'].std())]
    
def pH(data):
    
    return data[(data['pH']>=data['pH'].mean() -3*data['pH'].std()) &
                (data['pH']<= data['pH'].mean()+ 3*data['pH'].std())]
    
def sulphates(data):
    
    return data[(data['sulphates']>=data['sulphates'].mean() -3*data['sulphates'].std()) &
                (data['sulphates']<= data['sulphates'].mean()+ 3*data['sulphates'].std())]
    
def alcohol(data):
    
    return data[(data['alcohol']>=data['alcohol'].median() -3*data['alcohol'].std()) &
                (data['alcohol']<= data['alcohol'].median()+ 3*data['alcohol'].std())]
    
    
def clean(data):
    data.dropna()
    data = fixed_acidity(data)
    data = volatile_acidity(data)
    data = citric_acid(data)
    data = residual_sugar(data)
    data = chlorides(data)
    data = free_sulfur_dioxide(data)
    data = total_sulfur_dioxide(data)
    data = density(data)
    data = pH(data)
    data = sulphates(data)
    data = alcohol(data)
    
    return data


def test(data):
    echantillon = pd.DataFrame(columns=['fixed acidity',
                                        'volatile acidity',
                                        'citric acid',
                                        'residual sugar',
                                        'chlorides',
                                        'free sulfur dioxide',
                                        'total sulfur dioxide',
                                        'density',
                                        'pH',
                                        'sulphates',
                                        'alcohol',
                                        'quality'])
    compteur1 = 0
    compteur2 = 0
    while compteur1 < len(data):
        if compteur1 % 10 == 9:
            echantillon.loc[compteur2] = data.iloc[compteur1]
            compteur2 += 1
        compteur1 += 1
    return echantillon

def entrainement(data):
    echantillon = pd.DataFrame(columns=['fixed acidity',
                                        'volatile acidity',
                                        'citric acid',
                                        'residual sugar',
                                        'chlorides',
                                        'free sulfur dioxide',
                                        'total sulfur dioxide',
                                        'density',
                                        'pH',
                                        'sulphates',
                                        'alcohol',
                                        'quality'])
    compteur1 = 0
    compteur2 = 0
    while compteur1 < len(data):
        if compteur1 % 10 != 9:
            echantillon.loc[compteur2] = data.iloc[compteur1]
            compteur2 += 1
        compteur1 += 1
    return echantillon

































    """
            
              data_mauvais = data_m.median(skipna = True)
    data_bon = data_b.median(skipna = True)
    
    frames = [data_mauvais,data_bon]
    
    return pd.concat(frames,axis = 1, sort = False)
    """     
            
    

    
