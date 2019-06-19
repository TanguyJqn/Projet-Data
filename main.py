#coding utf-8

"""
@authors: Tanguy Jennequin, Maxime Labure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn as sci
from scipy.integrate import quad
from sklearn import discriminant_analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

def test_perceptron(data,limite = 100,learning_rate = 0.01):
    
    train = entrainement(data)   # recuperation de 90% des valeurs pour les donnees d'entrainement
    
    d_train = train.iloc[:,:11]         # on tronque les donnees en 2 -> le dataframe entier sans la qualite
    d_quality_train = train.iloc[:,11:]     # seulement la qualite ici
    
    d_numpy_train = np.array(d_train)               # conversion des donnees en tableau numpy pour pouvoir les traiter dans le perceptron
    d_numpy_quality_train = np.array(d_quality_train)
    
    
    
    evaluate = test(data)       # recuperation de 10% des valeurs pour les donnees de test
    
    data_test = evaluate.iloc[:,:11]         # on tronque les donnees en 2 -> le dataframe entier sans la qualite
    data_test_quality = evaluate.iloc[:,11:]  # seulement la qualite ici
    
    data_test_np = np.array(data_test)           # conversion des donnees en tableau numpy pour pouvoir les traiter dans le perceptron
    data_test_quality_np = np.array(data_test_quality)


    percep = Perceptron(11,limite,learning_rate)  # on initialise le perceptron avec les parametres de colonne, de seuil et de taux d'apprentissage
    percep.train(d_numpy_train,d_numpy_quality_train)    #apprentissage avec donnees d'entrainement
    accurancy = percep.perceptron_accurancy(data_test_np,data_test_quality_np) #appel de la fonction accurancy pour avoir le % de bonne prediction ( avec les donnees de test cette fois)
    print('precision du perceptron: {}'.format(accurancy))
    return accurancy     


class Perceptron(object): # creation d'une classe perceptron et ses attributs : nb d'entrees, seuil et taux d'apprentissage

    def __init__(self, nb_inputs, seuil=100, learning_rate=0.01):
        self.seuil = seuil
        self.learning_rate = learning_rate
        self.poids = np.zeros(nb_inputs + 1) # poids est le biais / on cree une colonne de zeros (derniere colonne) pour initialiser les biais

    def predict(self, inputs):
        somme = np.dot(self.poids[0:11],inputs.T) + self.poids[-1] #f(x) =1 si poids.inputs + biais > 0 sinon 0
                                        
        if somme > 0:  

            activation = 1   # vin de qualite, le label(resultat veridique) est de 1

        else:
            activation = -1  # vin de mauvaise qualite, le label (resultat veridique) est de -1

        return activation

    def train(self, training_inputs, labels):

        for lim in range(self.seuil): # declaration de la boucle qui permettra de faire 'seuils' fois l'entrainement
         
            for inputs, label in zip(training_inputs, labels): #inputs et label prennent respectivement les valeurs training_inputs et labels
   
                prediction = self.predict(inputs)   # on predit le resultat pour chaque ligne
                self.poids[0:11] += self.learning_rate * (label - prediction) * (inputs)  # ajustement du coefficient directeur de l'hyperplan

                self.poids[-1] += self.learning_rate * (label - prediction) # ajustement du biais ( ordonnee a l'origine)

                
    def perceptron_accurancy(self, data, label):

        nb_test = 0.0            # les deux variables sont initialisees de cette maniere pour eviter les conflits de type
        erreur = 0.0
        nb_test = data.shape[0]      #data.shape[0] recupere le nombre de ligne du data_test

        for row, resultat in zip(data, label):
            prediction = self.predict(row)

            if prediction != resultat:   # si la prediction est differente du resultat, alors on incremente l'erreur
                erreur += 1  # increments nb_error

        return (1 - erreur/nb_test) 

def start():
    data = pd.read_csv("projetData/red_wines.csv")
    data=clean(data)
    regressionLogistique(data)
    analyseDiscriminanteLineaire(data)
    analyseDiscriminanteQuadratique(data)
    svm(data)
    voisins(data)
    arbre(data)
    test_perceptron(data)
    return data

"""
Les 11 fonctions ci-dessous permettent d'epurer les valeurs de telles maniere a ce que l'on enleve les valeurs
qui sont 3x superieures ou inferieures a la moyenne ou la mediane.

Pour choisir entre la moyenne et la mediane, nous avons etabli quel attribut avait une allure gausienne ( et donc nous 
appliquons la moyenne) et quel attribut ne l'avait pas ( et donc la mediane cette fois-ci) 

"""

def fixed_acidity(data):
    return data[(data['fixed acidity'] >= data['fixed acidity'].mean() - 3 * data['fixed acidity'].std()) &
                (data['fixed acidity'] <= data['fixed acidity'].mean() + 3 * data['fixed acidity'].std())]

def volatile_acidity(data):
    return data[(data['volatile acidity'] >= data['volatile acidity'].mean() - 3 * data['volatile acidity'].std()) &
                (data['volatile acidity'] <= data['volatile acidity'].mean() + 3 * data['volatile acidity'].std())]

def citric_acid(data):
    return data[(data['citric acid'] >= data['citric acid'].median() - 3 * data['citric acid'].std()) &
                (data['citric acid'] <= data['citric acid'].median() + 3 * data['citric acid'].std())]

def residual_sugar(data):
    return data[(data['residual sugar'] >= data['residual sugar'].mean() - 3 * data['residual sugar'].std()) &
                (data['residual sugar'] <= data['residual sugar'].mean() + 3 * data['residual sugar'].std())]

def chlorides(data):
    return data[(data['chlorides'] >= data['chlorides'].mean() - 3 * data['chlorides'].std()) &
                (data['chlorides'] <= data['chlorides'].mean() + 3 * data['chlorides'].std())]

def free_sulfur_dioxide(data):
    return data[
        (data['free sulfur dioxide'] >= data['free sulfur dioxide'].median() - 3 * data['free sulfur dioxide'].std()) &
        (data['free sulfur dioxide'] <= data['free sulfur dioxide'].median() + 3 * data['free sulfur dioxide'].std())]

def total_sulfur_dioxide(data):
    return data[(data['total sulfur dioxide'] >= data['total sulfur dioxide'].median() - 3 * data[
        'total sulfur dioxide'].std()) &
                (data['total sulfur dioxide'] <= data['total sulfur dioxide'].median() + 3 * data[
                    'total sulfur dioxide'].std())]

def density(data):
    return data[(data['density'] >= data['density'].mean() - 3 * data['density'].std()) &
                (data['density'] <= data['density'].mean() + 3 * data['density'].std())]

def pH(data):
    return data[(data['pH'] >= data['pH'].mean() - 3 * data['pH'].std()) &
                (data['pH'] <= data['pH'].mean() + 3 * data['pH'].std())]

def sulphates(data):
    return data[(data['sulphates'] >= data['sulphates'].mean() - 3 * data['sulphates'].std()) &
                (data['sulphates'] <= data['sulphates'].mean() + 3 * data['sulphates'].std())]

def alcohol(data):
    return data[(data['alcohol'] >= data['alcohol'].median() - 3 * data['alcohol'].std()) &
                (data['alcohol'] <= data['alcohol'].median() + 3 * data['alcohol'].std())]

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
    data = data.drop_duplicates()
    #data.drop(['pH', 'alcohol', 'citric acid', 'total sulfur dioxide'], axis='columns', inplace=True)
    return data

def test(data):       # recuperation de 10% des valeurs ( une toute les 10 lignes) ce qui fera les donnees de train
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

def entrainement(data):     # recuperation de 90% des valeurs ( une toute les 10 lignes) ce qui fera les donnees de test
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

def corr(data):
    '''
        Function that implements the display of correlations between all attributes

       	:param data: The dataset that we use to train and test the classifier.
    '''
    fig=plt.figure()

    #creation du thermometre qui sert de legende
    ax=fig.add_subplot(111)
    c=ax.matshow(data.corr(), cmap=cm.jet)
    fig.colorbar(c)

    #graph principal
    ticks=np.arange(0, 12, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns.to_list())
    plt.xticks(rotation=90)
    ax.set_yticklabels(data.columns.to_list())
    plt.title('Correlations')
    ax.grid(False)

def nuages(data, attribute):
    '''
        Function that implements the scatter for the classe selected as parameter

       	:param data: The dataset that we use to train and test the classifier.
       	:param attribute: The attribute to use as abciss for the scatters
    '''
    plt.figure()
    for x in range(0, 11): #parcours de tous les attributs
        data.plot(kind='scatter', x=x, y=attribute)
    plt.show()

def draw_hist(data):
    '''
        Function that implements the display of the distribution of each attribute

       	:param data: The dataset that we use to train and test the classifier.
    '''
    data.hist(bins=100)

def draw_all(data):
    '''
        Function that implements the display of the histograms and scatters

       	:param data: The dataset that we use to train and test the classifier.
    '''
    from pandas.plotting import scatter_matrix
    scatter_matrix(data)
    plt.show()

def regressionLogistique(data):
    '''
        Function that implements the logistic regression classifier

    	:param data: The dataset that we use to train and test the classifier.
    '''
    #separation du jeu de donnees
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)

    #creation et entrainement du modele
    rl = LogisticRegression()
    rl.fit(X_train, Y_train)

    #eval training
    y_train_predict = rl.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)
    accurate = accuracy_score(Y_train, y_train_predict.round(), normalize=False)
    prec = precision_score(Y_train, y_train_predict.round(), average=None)
    f1 = f1_score(Y_train, y_train_predict.round(), average=None)

    affichage_scores('train regression logistique', quadratic_error, r2, accurate, prec, f1)

    #eval test
    y_test_predict = rl.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)
    accurate = accuracy_score(Y_test, y_test_predict.round(), normalize=False)
    prec = precision_score(Y_test, y_test_predict.round(), average=None)
    f1 = f1_score(Y_test, y_test_predict.round(), average=None)

    affichage_scores('test regression logistique', quadratic_error, r2, accurate, prec, f1)

def analyseDiscriminanteLineaire(data):
    '''
        Function that implements the linear discrimination analysis classifier

       	:param data: The dataset that we use to train and test the classifier.
    '''
    #separation du jeu de donnees
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)

    #creation et entrainement du modele
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)

    #eval training
    y_train_predict = lda.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)
    accurate = accuracy_score(Y_train, y_train_predict.round(), normalize=False)
    prec = precision_score(Y_train, y_train_predict.round(), average=None)
    f1 = f1_score(Y_train, y_train_predict.round(), average=None)

    affichage_scores('train analyse discrimante lineaire', quadratic_error, r2, accurate, prec, f1)

    #eval test
    y_test_predict = lda.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)
    accurate = accuracy_score(Y_test, y_test_predict.round(), normalize=False)
    prec = precision_score(Y_test, y_test_predict.round(), average=None)
    f1 = f1_score(Y_test, y_test_predict.round(), average=None)

    affichage_scores('test analyse discriminante lineaire', quadratic_error, r2, accurate, prec, f1)

def analyseDiscriminanteQuadratique(data):
    '''
        Function that implements the quadratic disciminant analysis classifier

        :param data: The dataset that we use to train and test the classifier.
    '''
    #separation du jeu de donnees
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)

    #creation et entrainement du modele
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, Y_train)

    #eval training
    y_train_predict = qda.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)
    accurate = accuracy_score(Y_train, y_train_predict.round(), normalize=False)
    prec = precision_score(Y_train, y_train_predict.round(), average=None)
    f1 = f1_score(Y_train, y_train_predict.round(), average=None)

    affichage_scores('train analyse discriminante quadratique', quadratic_error, r2, accurate, prec, f1)

    #eval test
    y_test_predict = qda.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)
    accurate = accuracy_score(Y_test, y_test_predict.round(), normalize=False)
    prec = precision_score(Y_test, y_test_predict.round(), average=None)
    f1 = f1_score(Y_test, y_test_predict.round(), average=None)

    affichage_scores('test analyse discriminante quadratique', quadratic_error, r2, accurate, prec, f1)

def svm(data):
    '''
        Function that implements the svm classifier

        :param data: The dataset that we use to train and test the classifier.
    '''
    #separation du jeu de donnees
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)

    #creation et entrainement du modele
    prd = SVC()
    prd.fit(X_train, Y_train)

    #eval training
    y_train_predict = prd.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)
    accurate = accuracy_score(Y_train, y_train_predict.round(), normalize=False)
    prec = precision_score(Y_train, y_train_predict.round(), average=None)
    f1 = f1_score(Y_train, y_train_predict.round(), average=None)

    affichage_scores('train svm', quadratic_error, r2, accurate, prec, f1)

    #eval test
    y_test_predict = prd.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)
    accurate = accuracy_score(Y_test, y_test_predict.round(), normalize=False)
    prec = precision_score(Y_test, y_test_predict.round(), average=None)
    f1 = f1_score(Y_test, y_test_predict.round(), average=None)

    affichage_scores('test svm', quadratic_error, r2, accurate, prec, f1)

def voisins(data):
    '''
        Function that implements the k-neighbours classifier

        :param data: The dataset that we use to train and test the classifier.
    '''
    #separation du jeu de donnees
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)

    #creation et entrainement du modele
    vois = KNeighborsClassifier()
    vois.fit(X_train, Y_train)

    #eval training
    y_train_predict = vois.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)
    accurate = accuracy_score(Y_train, y_train_predict.round(), normalize=False)
    prec = precision_score(Y_train, y_train_predict.round(), average=None)
    f1 = f1_score(Y_train, y_train_predict.round(), average=None)

    affichage_scores('train voisins', quadratic_error, r2, accurate, prec, f1)

    #eval test
    y_test_predict = vois.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)
    accurate = accuracy_score(Y_test, y_test_predict.round(), normalize=False)
    prec = precision_score(Y_test, y_test_predict.round(), average=None)
    f1 = f1_score(Y_test, y_test_predict.round(), average=None)

    affichage_scores('test voisins', quadratic_error, r2, accurate, prec, f1)

def arbre(data):
    '''
        Function that implements the decision tree classifier

        :param data: The dataset that we use to train and test the classifier.
    '''
    #separation du jeu de donnees
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)

    #creation et entrainement du modele
    abr = DecisionTreeClassifier()
    abr.fit(X_train, Y_train)

    #eval training
    y_train_predict = abr.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)
    accurate = accuracy_score(Y_train, y_train_predict.round(), normalize=False)
    prec = precision_score(Y_train, y_train_predict.round(), average=None)
    f1 = f1_score(Y_train, y_train_predict.round(), average=None)

    affichage_scores('train arbre', quadratic_error, r2, accurate, prec, f1)

    #eval test
    y_test_predict = abr.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)
    accurate = accuracy_score(Y_test, y_test_predict.round(), normalize=False)
    prec = precision_score(Y_test, y_test_predict.round(), average=None)
    f1 = f1_score(Y_test, y_test_predict.round(), average=None)

    affichage_scores('test arbre', quadratic_error, r2, accurate, prec, f1)

def affichage_scores(type, quadratic_error, r2, accurate, prec, f1):
    '''
        display of the score of a train or a test of a classifier

       	:param type: 'train' or 'test' + classifier model
       	:param quadratic_error: float
       	:param r2: float
       	:param accurate: int
       	:param prec: array of float
       	:param f1: array of float
    '''
    print('performances {}'.format(type))
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))
    print('accuracy score: {}'.format(accurate))
    print('precision: {}'.format(prec))
    print('score f1: {}'.format(f1))
    print('\n')