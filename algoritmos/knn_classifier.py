import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Imports de sklearn somente para dividir treino e teste e visualizar o resultado
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Definição do KNN-Classifier
def knn_classifier(X_train, y_train, X_test, k):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)

    y_pred = []

    for i in range(len(X_test)):
        current = X_test[i]

        #Cálculo das distâncias entre o ponto atual e todos os pontos de treino
        distances = np.sqrt(np.sum((current - X_train)**2, axis=1))

        #Determinação dos k vizinhos mais próximos
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]

        #Escolha da classe mais frequente entre os vizinhos
        values, counts = np.unique(nearest_labels, return_counts=True)
        predicted_label = values[np.argmax(counts)]

        y_pred.append(predicted_label)

    return np.array(y_pred)

#Definição do dataframe Iris Flowers
df = pd.read_csv('datasets/IRIS.csv')
X = df.drop(columns=['species'])
y = df['species']

#Divisão entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Aplicação do KNN-Classifier
y_pred = knn_classifier(X_train, y_train, X_test, k=75)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y.unique())
disp.plot(cmap='Blues')
plt.title(f'Matriz de Confusão - KNN (Iris Flowers)')
plt.show()
