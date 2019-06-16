#coding utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn as sci
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

def start():
    data = pd.read_csv("projetData/red_wines.csv")
    data=clean(data)
    return data

def mauvais(data):
    return data[data['quality'] == -1]

def bon(data):
    return data[data['quality'] == 1]

def median(data_m, data_b):
    data_mauvais = data_m.median(skipna=True)
    data_bon = data_b.median(skipna=True)

    frames = [data_mauvais, data_bon]

    return pd.concat(frames, axis=1, sort=False)

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
    data.drop(['pH', 'alcohol', 'citric acid', 'total sulfur dioxide'], axis='columns', inplace=True)
    return data

def echantillon10(data):
    echantillon = pd.DataFrame(columns=['fixed acidity',
                                'volatile acidity',
                                'residual sugar',
                                'chlorides',
                                'free sulfur dioxide',
                                'density',
                                'sulphates',
                                'quality'])
    compteur1 = 0
    compteur2 = 0
    while compteur1 < len(data):
        if compteur1 % 10 == 9:
            echantillon.loc[compteur2] = data.iloc[compteur1]
            compteur2 += 1
        compteur1 += 1
    return echantillon

def echantillon90(data):
    echantillon = pd.DataFrame(columns=['fixed acidity',
                                        'volatile acidity',
                                        'residual sugar',
                                        'chlorides',
                                        'free sulfur dioxide',
                                        'density',
                                        'sulphates',
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
    fig=plt.figure()
    ax=fig.add_subplot(111)
    c=ax.matshow(data.corr(), cmap=cm.jet)
    fig.colorbar(c)
    ticks=np.arange(0, 8, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns.to_list())
    plt.xticks(rotation=90)
    ax.set_yticklabels(data.columns.to_list())
    plt.title('Correlations')
    ax.grid(False)

def nuages(data):
    plt.figure()
    for x in range(0, 11):
        data.plot(kind='scatter', x=x, y=11)
    plt.show()

def draw_hist(data):
    data.hist(bins=100)

def draw_all(data):
    from pandas.plotting import scatter_matrix
    scatter_matrix(data)
    plt.show()

def regressionLineaire(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    lml = LinearRegression()
    lml.fit(X_train, Y_train)

    #eval training
    y_train_predict = lml.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)

    print('perf train')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

    #eval test
    y_test_predict = lml.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)

    print('perf test')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

def regressionLogistique(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    rl = LogisticRegression()
    rl.fit(X_train, Y_train)

    #eval training
    y_train_predict = rl.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)

    print('perf train')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

    #eval test
    y_test_predict = rl.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)

    print('perf test')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

def analyseDiscriminanteLineaire(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)

    #eval training
    y_train_predict = lda.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)

    print('perf train')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

    #eval test
    y_test_predict = lda.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)

    print('perf test')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

def analyseDiscriminanteQuadratique(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, Y_train)

    #eval training
    y_train_predict = qda.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)

    print('perf train')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

    #eval test
    y_test_predict = qda.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)

    print('perf test')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

def svm(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    prd = SVC()
    prd.fit(X_train, Y_train)

    #eval training
    y_train_predict = prd.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)

    print('perf train')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

    #eval test
    y_test_predict = prd.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)

    print('perf test')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

def voisins(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    vois = KNeighborsClassifier()
    vois.fit(X_train, Y_train)

    #eval training
    y_train_predict = vois.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)

    print('perf train')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

    #eval test
    y_test_predict = vois.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)

    print('perf test')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

def arbre(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data[data.columns[:-1]], data['quality'], test_size=0.2, random_state=5)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    abr = DecisionTreeClassifier()
    abr.fit(X_train, Y_train)

    #eval training
    y_train_predict = abr.predict(X_train)
    quadratic_error = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    r2 = r2_score(Y_train, y_train_predict)

    print('perf train')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))

    #eval test
    y_test_predict = abr.predict(X_test)
    quadratic_error = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2 = r2_score(Y_test, y_test_predict)

    print('perf test')
    print('err quadra moy: {}'.format(quadratic_error))
    print('score r2: {}'.format(r2))
