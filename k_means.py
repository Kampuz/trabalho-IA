import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

#Definição do K-Means
def KMeans(X, y, k, max_iter=100, tol=1e-6):
    data = np.column_stack((X, y))
    n_samples = data.shape[0]

    #Inicialização dos centróides
    indices = np.random.choice(n_samples, size=k, replace=False)
    centroids = data[indices, :]

    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        #Atribuição dos pontos aos clusters
        for i in range(n_samples):
            distances = np.sqrt(np.sum((data[i] - centroids)**2, axis=1))
            labels[i] = np.argmin(distances)

        #Atualização dos centróides
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                new_centroids[c] = data[mask].mean(axis=0)
            else:
                new_centroids[c] = centroids[c]

        #Verificação de convergência
        shift = np.sqrt(np.sum((new_centroids - centroids)**2))
        centroids = new_centroids
        if shift < tol:
            break

    return centroids, labels, data

#Definição do dataframe Iris Flowers
df = pd.read_csv('IRIS.csv')
X = df['sepal_length'].values
y = df['sepal_width'].values

#Aplicação do K-Means
centroids, labels, data = KMeans(X, y, k=3)

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolors='k', label='Amostras')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centróides')
plt.title('K-Means Clustering (Iris Flowers)')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()
plt.show()
