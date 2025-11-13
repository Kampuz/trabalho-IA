import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def add_offset(X):
    n_samples = X.shape[0]
    n_features = X.shape[1]

    X_with_offset = np.zeros((n_samples, n_features + 1))

    for i in range(n_samples):
        X_with_offset[i, 0] = 1.0
    
    for i in range(n_samples):
        for j in range(n_features):
            X_with_offset[i, j + 1] = X[i, j]

    X = X_with_offset

    return X

# normaliza as colunas de características (0 a 1)
def normalize_x(X):
    X = (X - X.mean()) / X.std()
    return X

# function to prepare X (normalize + add a offset)
def prepare_X(X):
    return add_offset(normalize_x(X))

# map every possible result in a class number
def prepare_Y(y):
    mapping = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }

    return y.map(mapping)

# Calculate the prob of a sample belonging to a class
def sigmoid(z):
     return 1/ (1 + np.exp(-z))
    
# tests a class against all others
def logistic_regresion_binary(X, y, learning_rate=0.1, epochs=1000):
    weights = np.zeros(X.shape[1])

    for i in range(epochs):
        z = np.dot(X, weights)
        predictions = sigmoid(z)

        # calcs the gradient
        gradient = np.dot(X.T, (predictions - y)) / len(y)
        
        # update weights using descending gradient
        weights -= learning_rate * gradient
    
    return weights

def logistic_regression_multiclass(X_train, y_train, X_test, learning_rate=0.1, epochs=1000):
    classes = np.unique(y_train)
    all_weights = []


    #for every class creates a model where 1 is current class and 0 is another class
    for c in classes:
        y_binary = (y_train == c).astype(int)
        weights = logistic_regresion_binary(X_train, y_binary, learning_rate, epochs)
        all_weights.append(weights)
    
    # combinates all the weights, calculates the score and the prob
    all_weights = np.array(all_weights)
    scores = np.dot(X_test, all_weights.T)
    probs = sigmoid(scores)

    y_pred = np.argmax(probs, axis=1)
    return y_pred


# Definição do dataframe iris Flowers
df = pd.read_csv('IRIS.csv')

X = prepare_X(df.drop(columns=['species']))
y = prepare_Y(df['species'])

# Divisão entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicação da Regressão Lógica
y_pred = logic_regression_multiclass(X_train, y_train, X_test, 0.1, 1000)

# Exibição de uma matriz de confusão com os resultados
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y.unique())
disp.plot(cmap='Blues')
plt.title(f'Matriz de Confusão - Regressão Lógica (Iris Flowers)')
plt.show()
