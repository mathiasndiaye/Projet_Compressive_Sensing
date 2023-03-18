import numpy as np
from sklearn.decomposition import DictionaryLearning
import pandas as pd 


# Charger les données d'apprentissage à partir du fichier "data"
df = pd.read_excel('data.xlsx', header= 0, nrows=100)
data = df.to_numpy()

# Définir les paramètres de l'apprentissage du dictionnaire
n_components = 100  # nombre de colonnes dans le dictionnaire
n_iter = 10  # nombre d'itérations de l'algorithme K-SVD
alpha = 1  # coefficient de régularisation L1
tol = 1e-6  # tolérance pour l'arrêt de l'algorithme

# Créer une instance de la classe DictionaryLearning et lancer l'apprentissage
dico_learner = DictionaryLearning(n_components=n_components, alpha=alpha, tol=tol, max_iter=n_iter)
dico_learner.fit(data.T)  # transposer les données pour avoir les vecteurs d'apprentissage en colonnes

# Obtenir le dictionnaire appris
D = dico_learner.components_.T  # transposer pour avoir les atomes en colonnes

# Afficher le dictionnaire appris
print(D.shape)
