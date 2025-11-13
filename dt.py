import numpy as np
import mlcroissant as mlc
import pandas as pd
import math


class Node:
    def __init__(self, atributo=None, threshold=None, left=None, right=None, value=None):
        self.atributo = atributo
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, x, y, atributos):
        self.root = self.dt(x,y,atributos,self.moda(y))

    def moda(self, y):
        n = len(y)
        n0 = n1 = n2 = 0
        for i in range(n):
            if y[i] == 0:
                n0 += 1
            if y[i] == 1:
                n1 += 1
            if y[i] == 2:
                n2 += 1
        maximo = max(n0,n1,n2)
        if(maximo == n0):
            return 0
        if(maximo == n1):
            return 1
        if(maximo == n2):
            return 2
        return 9


    def h(self, y):
        Y = np.array(y)
        n = len(Y)
        if n == 0:
            return 0

        entropia = 0
        n0 = n1 = n2 = 0


        for i in range(n):
            if Y[i] == 0:
                n0 += 1
            if Y[i] == 1:
                n1 += 1
            if Y[i] == 2:
                n2 += 1

        p0 = n0 / n
        p1 = n1 / n
        p2 = n2 / n

        probabilidades = [p0,p1,p2]

        for p in probabilidades:
            if p > 0 :
                entropia += p * math.log2(p)

        return -entropia

    def ig(self, y, y_left, y_right):
        parent_entropy = self.entropia(y)

        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        if n == 0:
            return 0

        child_entropy = (n_left / n) * self._entropy(y_left) + (n_right / n) * self._entropy(y_right)

        return parent_entropy - child_entropy


    def escolherAtributo(self, atributos, X, y):
        igMax = -1
        atributoEscolhido = None
        valorEscolhido = None

        n = len(X)
        if n == 0:
            return None, None

        entropia_atual = self.h(y)

        if atributos is None:
            return None, None

        for atributo in atributos:
            valores = x[atributo]

            for valor in valores:
                esq_y = [y[i] for i in range(n) if X[atributo][i] <= valor]
                dir_y = [y[i] for i in range(n) if X[atributo][i] > valor]

                esq_n , dir_n = len(esq_y), len(dir_y)

                ganho = entropia_atual - (esq_n/n * self.h(esq_y) + (dir_n/n * self.h(dir_y)))

                if ganho > igMax:
                    igMax = ganho
                    atributoEscolhido = atributo
                    valorEscolhido = valor

        return atributoEscolhido, valorEscolhido


    def dt(self, x, y, atributos, padrao):
        if len(x) == 0:
            return Node(value=padrao)
        if atributos is None:
            return Node(value=self.moda(y))

        atributo, valor = self.escolherAtributo(atributos, x, y)

        esq_X, esq_y = [], []
        dir_X, dir_y = [], []

        for i in range(len(x)):
            if x[atributo][i] <= valor:
                esq_X.append(x[atributo][i])
                esq_y.append(y[i])
            else:
                dir_X.append(x[atributo][i])
                dir_y.append(y[i])

        esq_atributos = dir_atributos = atributos.remove(atributo)
        esq_tree = self.dt(esq_X, esq_y, esq_atributos,self.moda(y))
        dir_tree = self.dt(dir_X, dir_y, dir_atributos,self.moda(y))

        return Node(atributo=atributo, threshold=valor, left=esq_tree, right=dir_tree)

    def getAtributo(self, atributo):
        if atributo == 'sepal_length':
            return 0
        if atributo == 'sepal_width':
            return 1
        if atributo == 'petal_length':
            return 2
        if atributo == 'petal_width':
            return 3

    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value

        i = self.getAtributo(node.atributo)
        if x[i] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in np.array(X)])


df = pd.read_csv('IRIS.csv')

x = df[ ['sepal_length','sepal_width','petal_length','petal_width'] ]
atributos = ['sepal_length','sepal_width','petal_length','petal_width']
thisdict = {
        'Iris-setosa' : 0,
        'Iris-versicolor' : 1 ,
        'Iris-virginica' : 2
    }
y = df['species'].map(thisdict)

dt = DecisionTree(x,y,atributos)

teste = dt.predict(x)

print(teste)
