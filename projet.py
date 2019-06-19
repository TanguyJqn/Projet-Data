# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:50:12 2019

@author: Tanguy Jennequin
"""
import pandas as pd
import numpy as np


def test_perceptron(data,limite = 4000,learning_rate = 0.01):
    
    train = entrainement(data)   # recupération de 90% des valeurs pour les données d'entrainement
    
    d_train = train.iloc[:,:11]         # on tronque les données en 2 -> le dataframe entier sans la qualité
    d_quality_train = train.iloc[:,11:]     # seulement la qualité ici
    
    d_numpy_train = np.array(d_train)               # conversion des données en tableau numpy pour pouvoir les traiter dans le perceptron
    d_numpy_quality_train = np.array(d_quality_train)
    
    
    
    evaluate = test(data)       # recupération de 10% des valeurs pour les données de test
    
    data_test = evaluate.iloc[:,:11]         # on tronque les données en 2 -> le dataframe entier sans la qualité
    data_test_quality = evaluate.iloc[:,11:]  # seulement la qualité ici
    
    data_test_np = np.array(data_test)           # conversion des données en tableau numpy pour pouvoir les traiter dans le perceptron
    data_test_quality_np = np.array(data_test_quality)


    percep = Perceptron(11,limite,learning_rate)  # on initialise le perceptron avec les paramètres de colonne, de seuil et de taux d'apprentissage
    percep.train(d_numpy_train,d_numpy_quality_train)    #apprentissage avec données d'entrainement
    accurancy = percep.perceptron_accurancy(data_test_np,data_test_quality_np) #appel de la fonction accurancy pour avoir le % de bonne prédiction ( avec les données de test cette fois) 

    return accurancy     


class Perceptron(object): # création d'une classe perceptron et ses attributs : nb d'entrées, seuil et taux d'apprentissage

    def __init__(self, nb_inputs, seuil=4000, learning_rate=0.01):
        self.seuil = seuil
        self.learning_rate = learning_rate
        self.poids = np.zeros(nb_inputs + 1) # poids est le biais / on crée une colonne de zéros (derniere colonne) pour initialiser les biais

    def predict(self, inputs):
        somme = np.dot(self.poids[0:11],inputs.T) + self.poids[-1] #f(x) =1 si poids.inputs + biais > 0 sinon 0
                                        
        if somme > 0:  

            activation = 1   # vin de qualité, le label(résultat véridique) est de 1

        else:
            activation = -1  # vin de mauvaise qualité, le label (résultat véridique) est de -1

        return activation

    def train(self, training_inputs, labels):

        for lim in range(self.seuil): # déclaration de la boucle qui permettra de faire 'seuils' fois l'entrainement
         
            for inputs, label in zip(training_inputs, labels): #inputs et label prennent respectivement les valeurs training_inputs et labels
   
                prediction = self.predict(inputs)   # on prédit le résultat pour chaque ligne
                self.poids[0:11] += self.learning_rate * (label - prediction) * (inputs)  # ajustement du coefficient directeur de l'hyperplan

                self.poids[-1] += self.learning_rate * (label - prediction) # ajustement du biais ( ordonnée à l'origine)

                
    def perceptron_accurancy(self, data, label):

        nb_test = 0.0            # les deux variables sont initialisées de cette manière pour éviter les conflits de type
        erreur = 0.0
        nb_test = data.shape[0]      #data.shape[0] récupère le nombre de ligne du data_test

        for row, resultat in zip(data, label):
            prediction = self.predict(row)

            if prediction != resultat:   # si la prédiction est différente du résultat, alors on incrémente l'erreur
                erreur += 1  # increments nb_error
        
        return (1 - erreur/nb_test) 

      

def start():
    data = pd.read_csv("red_wines.csv")
    data = clean(data)
    return data

"""
Les 11 fonctions ci-dessous permettent d'épurer les valeurs de telles manière à ce que l'on enlève les valeurs
qui sont 3x supérieures ou inférieures à la moyenne ou la médiane.

Pour choisir entre la moyenne et la médiane, nous avons établi quel attribut avait une allure gausienne ( et donc nous 
appliquons la moyenne) et quel attribut ne l'avait pas ( et donc la médiane cette fois-ci) 


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


def test(data):       # récupération de 10% des valeurs ( une toute les 10 lignes) ce qui fera les données de test                                   
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

def entrainement(data):     # récupération de 90% des valeurs ( une toute les 10 lignes) ce qui fera les données de test  
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
