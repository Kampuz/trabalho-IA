import numpy as np
import pandas as pd
import math

class Node:
    def __init__(self, atributo=None, threshold=None, esquerda=None, direita=None, valor=None):
        self.atributo = atributo
        self.threshold = threshold
        self.esquerda = esquerda
        self.direita = direita
        self.valor = valor

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
        n = len(y)
        if n == 0:
            return 0

        entropia = 0
        n0 = n1 = n2 = 0


        for i in range(n):
            if y[i] == 0:
                n0 += 1
            if y[i] == 1:
                n1 += 1
            if y[i] == 2:
                n2 += 1

        p0 = n0 / n
        p1 = n1 / n
        p2 = n2 / n

        probabilidades = [p0,p1,p2]

        for p in probabilidades:
            if p > 0 :
                entropia -= p * math.log2(p)

        return entropia


    def escolherAtributo(self, atributos, x, y):
        igMax = -1
        atributoEscolhido = None
        valorEscolhido = None

        n = len(x)
        if n == 0:
            return None, None

        entropia_atual = self.h(y)

        if not atributos:
            return None, None

        for atributo in atributos:
            valores = x[atributo]

            for valor in valores:
                esq_y, dir_y = [], []
                for i in range(0,n):
                    if x[atributo][i] <= valor:
                        esq_y.append(y[i])
                    else:
                        dir_y.append(y[i])

                esq_n , dir_n = len(esq_y), len(dir_y)

                ganho = entropia_atual - (esq_n/n * self.h(esq_y) + (dir_n/n * self.h(dir_y)))
                if ganho > igMax:
                    igMax = ganho
                    atributoEscolhido = atributo
                    valorEscolhido = valor

        return atributoEscolhido, valorEscolhido

    def atributosIguais(self, y):
        aux = y[0]
        for i in y:
            if aux != i:
                return False
        return True

    def remover(self, atributos, atributo):
        aux = []
        for i in atributos:
            if i != atributo:
                aux.append(i)
        return aux


    def dt(self, x, y, atributos, padrao):
        if len(x) <= 2:
            return Node(valor=padrao)
        if atributos is None:
            return Node(valor=self.moda(y))
        if self.atributosIguais(y):
            return Node(valor=self.moda(y))

        atributo, valor = self.escolherAtributo(atributos, x, y)
        if atributo is None:
            return Node(valor = padrao)
        esq_x = pd.DataFrame(columns=atributos)
        dir_x = pd.DataFrame(columns=atributos)

        esq_y = dir_y = []

        for i in range(len(x)):
            if x[atributo][i] <= valor:
                esq_x.loc[len(esq_x)] = x.iloc[i]
                if not esq_y:
                    esq_y = [y[i]]
                else:
                    esq_y.append(y[i])
            else:
                dir_x.loc[len(dir_x)] = x.iloc[i]
                if not dir_y:
                    dir_y = [y[i]]
                else:
                    dir_y.append(y[i])


        esq_atributos = dir_atributos = self.remover(atributos,atributo)

        del esq_x[atributo]
        del dir_x[atributo]

        esq_tree = self.dt(esq_x, esq_y, esq_atributos,self.moda(y))
        dir_tree = self.dt(dir_x, dir_y, dir_atributos,self.moda(y))

        return Node(atributo=atributo, threshold=valor, esquerda=esq_tree, direita=dir_tree)

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
        if node.valor is not None:
            return node.valor

        i = self.getAtributo(node.atributo)
        if x[i] <= node.threshold:
            return self._predict_single(x, node.esquerda)
        else:
            return self._predict_single(x, node.direita)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in np.array(X)])

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        indent = "  " * depth

        if node.valor is not None:
            print(f"{indent}Folha: Classe {node.valor}")
        else:
            print(f"{indent}Atributo {node.atributo} <= {node.threshold:.2f}")
            self.print_tree(node.esquerda, depth + 1)
            self.print_tree(node.direita, depth + 1)


df = pd.read_csv('IRIS.csv')

x = df[ ['sepal_length','sepal_width','petal_length','petal_width'] ]
atributos = ['sepal_length','sepal_width','petal_length','petal_width']
dicionario = {
        'Iris-setosa' : 0,
        'Iris-versicolor' : 1 ,
        'Iris-virginica' : 2
    }
y = (df['species'].map(dicionario))
y.tolist()

dt = DecisionTree(x,y,atributos)
teste = dt.predict(x)
print(teste)

dt.print_tree()

n = len(x)
acertos = 0
for i in range(n):
    if teste[i] == y[i]:
        acertos += 1
print("Acuracia: ", acertos/n*100,"%")
