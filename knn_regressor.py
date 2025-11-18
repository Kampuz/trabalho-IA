import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# distancia euclidiana entre dois pontos/vetores
def measure_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

# retorna o elemento de uma tupla na posicao pos
def get_tuple(x, pos=0):
    return x[pos]

# encontra os k vizinhos mais próximos de x dentro de x_array
def get_k_near_neighbour(x, x_array, y_array, k=3):
    distancias = []
    for i in range(len(x_array)):
        d = measure_distance(x, x_array[i])
        distancias.append((d, y_array[i]))
    
    distancias_ordenadas = sorted(distancias, key=get_tuple)
    k_neighbours = [distancias_ordenadas[i] for i in range(k)]
    
    return k_neighbours

# calcula a média dos valores y dos vizinhos
def average_response(values):
    values_y = [v[1] for v in values]
    return sum(values_y)/len(values_y)

# prediz o valor de y para um ponto x usando knn
def predict_knn(x, x_array, y_array, k):
    neighbour = get_k_near_neighbour(x, x_array, y_array, k)
    return average_response(neighbour)

# plota a predição do knn para uma feature específica
def plot_knn_feature(x_array, y_array, feature_index, k):
    x_feature = x_array[:, feature_index]   # 1D
    y_array_1d = y_array.ravel()            # converte para 1D

    x_values = np.linspace(min(x_feature), max(x_feature), 200)
    y_pred = []

    for v in x_values:
        x_input = np.mean(x_array, axis=0).copy()
        x_input[feature_index] = v
        y_pred.append(predict_knn(x_input, x_array, y_array_1d, k))

    feature = ["sepal_length", "sepal_width", "petal_length"]
    plt.scatter(x_feature, y_array_1d)
    plt.plot(x_values, y_pred, color="red", label="KNN")
    plt.xlabel(f"Feature {feature[feature_index]}")
    plt.ylabel("Target: petal_width")
    plt.legend()
    plt.grid()
    plt.show()

# carrega o dataset iris
df = pd.read_csv("iris.csv")

sep_len = "sepal_length"
sep_wid = "sepal_width"
pet_len = "petal_length"
pet_wid = "petal_width"

# define features e target
x_array = df[[sep_len, sep_wid, pet_len]].values
y_array = df[[pet_wid]].values

# pontos de teste
x = np.array([5.1, 3.5, 0.4])
k = 3

# x = np.array([float(input("sepal length: ")), float(input("sepal width: ")), float(input("petal length: "))])
# k = int(input("K neighbours: "))

# plotagem
for i in range(3):
    plot_knn_feature(x_array, y_array, i, k)