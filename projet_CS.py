import numpy as np
from sklearn.decomposition import DictionaryLearning
import pandas as pd 


# APPRENTISSAGE DU DICTIONNAIRE VIA LA METHODE K-SVD

# Charger les données d'apprentissage à partir du fichier "data"
df = pd.read_excel('data.xlsx', header= 0, nrows=100)
data = df.to_numpy()

# Définir les paramètres de l'apprentissage du dictionnaire
n_components = 100  # nombre de colonnes dans le dictionnaire
n_iter = 10  # nombre d'itérations de l'algorithme K-SVD
alpha = 1  # coefficient de régularisation, plus il est grand plus il y aura de 0 dans le dico - 1 est le nb par défaut 
tol = 1e-6  # tolérance pour l'arrêt de l'algorithme

# Créer une instance de la classe DictionaryLearning et lancer l'apprentissage
dico_learner = DictionaryLearning(n_components=n_components, alpha=alpha, tol=tol, max_iter=n_iter)
dico_learner.fit(data.T)  # transposer les données pour avoir les vecteurs d'apprentissage en colonnes

# Obtenir le dictionnaire appris
D = dico_learner.components_.T  # transposer pour avoir les atomes en colonnes

# Afficher le dictionnaire appris
print(D)





#IMPLEMENTATION DES ALGOS DE CODAGE PARCIMONIEUX - FAIT EN TD 


#Implémentation StOMP - 200 itérations max et précision à 1e-6 et t doit être compris entre 2 et 3
def stOMP(X,D,t) :
    m, n = D.shape                #On récupère la taille de la matrice de notre dictionnaire (m lignes, n colonnes)
    R = X                         #On vient stocker notre signal X dans la variable R - La variable R constitue le résiduel de la différence entre norm(x) et norm(D*Alpha). On initialise R avec X
    alpha = np.zeros(n)           #On initialise notre vecteur parcimonieux alpha comme étant un vecteur constitués uniquement de 0 et de taille n (Afin que le produit matriciel D*alpha soit faisable) 
    K = 1                         #On initialise une variable K à 1 (car sinon on divise par 0 lors du calcul du seuillage) - Ici K est notre compteur, afin que l'algo ne tourne pas plus de N fois 
    P = []                        #On initialise une liste P vide - Ici P constituera la liste des positions des atomes sélectionnés
    res = []                      #On initialise une liste res vide - Son unique but est de stocker les resultats du calcul qu'on effectue afin de selectionner l'atome  
    while np.linalg.norm(R,2)>1e-6 and K<201:                 #On boucle tant que la norme 2 de R n'est pas inférieure à 1e-6 ou que l'on n'a pas dépassé le nombre max d'itérations (200)
        for i in range(n):                                     #On boucle sur chaque colonne (=atomes)
            d = D[:,i]                                        #On sélectionne la colonne en cours et la stocke dans d
            norm_d = np.linalg.norm(d,2)                      #On calcule sa norme 
            B = abs(np.transpose(d)@R)                        #On calcule la valeur absolue du produit scalaire entre d et R 
            res.append(B/norm_d)                              #On calcule le quotien : valeur absolue du produit scalaire entre d et R / norme de la colonne en cours. Et stocke le résultat dans la liste res qui constitue la liste des contributions 
        S=t*np.linalg.norm(R,2)/sqrt(K)                       #Calcul du seuillage
        Ak=[]                                                 #Initialisation de la liste vide Ak qui va venir stocker tous les inddices des atomes séléctionnés
        for contribution in res:                              #On boucle sur chaque contribution calculées 
            if contribution>S:                                #Si contribution est supérieur au seuillage
                Ak.append(res.index(contribution))            #Alors on l'ajoute à la liste Ak
        P=np.unique(P+Ak).tolist()                            #On ajoute a P les positions des atomes sélectionnés n'étant pas déjà dans la liste P
        phi=D[:,P]                                            #Phi est le dictionnaire actif, donc le dictionnaire D avec seulement les atomes ayant été sélectionnés
        alpha[P] = np.linalg.pinv(phi)@X                      #On vient mettre à jour uniquement les coefficient de Alpha contenue dans P donc ceux dont leurs atomes correspondants ont déjà été selectionné. On vient mettre à jour ces valeurs en résolvant le problème de minimisation norm(X-D*Alpha) par rapport à Alpha
        R = X-D@alpha                                         #On vient mettre à jour notre vecteur R qui nous sert à calculer le résiduel et à stopper l'algo si la précision voulue a été atteinte
        res = []                                              #On vient vider notre liste res afin de pouvoir stocker les nouveaux calculs à la prochaine itération
        K = K+1                                               #On incrémente K
    
    return [alpha, R, K, P]                                   #On retourne 4 éléments : Alpha (vecteur parcimonieux) - R (Vecteur résiduel) - K (Nombre d'itérations de l'algorithme) - P (Liste des index des atomes sélectionnés)


#Implémentation CoSaMP 
def CoSaMP(X,D,eps,N,s):
    m, n = D.shape                #On récupère la taille de la matrice de notre dictionnaire (m lignes, n colonnes)
    R = X                         #On vient stocker notre signal X dans la variable R - La variable R constitue le résiduel de la différence entre norm(x) et norm(D*Alpha). On initialise R avec X
    alpha = np.zeros(n)           #On initialise notre vecteur parcimonieux alpha comme étant un vecteur constitués uniquement de 0 et de taille n (Afin que le produit matriciel D*alpha soit faisable) 
    K = 0                         #On initialise une variable K à 1 (car sinon on divise par 0 lors du calcul du seuillage) - Ici K est notre compteur, afin que l'algo ne tourne pas plus de N fois 
    res = []                      #On initialise une liste res vide - Son unique but est de stocker les resultats du calcul qu'on effectue afin de selectionner l'atome  
    supp=[]                       #On initialise une list supp qui viendra stocker les supports
    while np.linalg.norm(R,2)>eps and K<N:                          #On boucle tant que la norme 2 de R n'est pas inférieure à 1e-6 ou que l'on n'a pas dépassé le nombre max d'itérations (200)
        for i in range(n):                                          #On boucle sur chaque colonne (=atomes)
            d = D[:,i]                                              #On sélectionne la colonne en cours et la stocke dans d
            norm_d = np.linalg.norm(d,2)                            #On calcule sa norme 
            B = abs(np.transpose(d)@R)                              #On calcule la valeur absolue du produit scalaire entre d et R 
            res.append(B/norm_d)                                    #On calcule le quotien : valeur absolue du produit scalaire entre d et R / norme de la colonne en cours. Et stocke le résultat dans la liste res qui constitue la liste des contributions 
        supp1 = np.argsort(res)[-(2*s):].tolist()                   #On vient stocker dans supp1 la liste des 2*s atomes avec la contribution la plus grande
        supp=np.unique(supp+supp1).tolist()                         #supp = supp ∪ supp1  
        AS=D[:,supp]                                                #AS est le dictionnaire actif contenant uniquement les 2*s atomes avec la plus grande contribution
        alpha[supp] = np.linalg.pinv(AS)@X                          #On vient mettre à jour uniquement les coefficient de Alpha contenue dans supp donc ceux dont leurs atomes correspondants ont déjà été selectionné. On vient mettre à jour ces valeurs en résolvant le problème de minimisation norm(X-D*Alpha) par rapport à Alpha
        alpha=garder_n_plus_grands_elements(alpha.tolist(),s)       #REJET - On garde seulement les s plus grands coefficients
        R = X-D@alpha                                               #On vient mettre à jour notre vecteur R qui nous sert à calculer le résiduel et à stopper l'algo si la précision voulue a été atteinte
        res = []                                                    #On vient vider notre liste res afin de pouvoir stocker les nouveaux calculs à la prochaine itération
        K = K+1                                                     #On incrémente K
    
    return [alpha, R, K, supp]                                   #On retourne 4 éléments : Alpha (vecteur parcimonieux) - R (Vecteur résiduel) - K (Nombre d'itérations de l'algorithme) - supp (Support, liste des index des atomes sélectionnés)   


