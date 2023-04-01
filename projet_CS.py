import numpy as np
from sklearn.decomposition import DictionaryLearning
import pandas as pd 
from math import sqrt
from sklearn.linear_model import orthogonal_mp_gram
from ApproximateKSVD import ApproximateKSVD


# APPRENTISSAGE DU DICTIONNAIRE VIA LA METHODE K-SVD

# Charger les données d'apprentissage à partir du fichier "data"
df = pd.read_excel('data.xlsx', header= 0, nrows=100)
data = df.to_numpy()
data=data.transpose()           #il faut transposer la matrice d'entrainement car notre fonction fit s'attends a recevoir n_samples en ligne et n_features en colonne

#Charger les vecteurs de test à partir du fichier "vecteursTest.xlsx"
df = pd.read_excel('vecteurTest.xlsx', header= 1, nrows=99)
vecteursTest= df.to_numpy()

"""
# Définir les paramètres de l'apprentissage du dictionnaire
n_components = 50  # nombre de colonnes dans le dictionnaire
n_iter = 200  # nombre d'itérations de l'algorithme K-SVD


alpha = 1  # coefficient de régularisation, plus il est grand plus il y aura de 0 dans le dico - 1 est le nb par défaut 
tol = 1e-6  # tolérance pour l'arrêt de l'algorithme

# Créer une instance de la classe DictionaryLearning et lancer l'apprentissage
dico_learner = DictionaryLearning(n_components=n_components, alpha=alpha, tol=tol, max_iter=n_iter)
dico_learner.fit(data.T)  # transposer les données pour avoir les vecteurs d'apprentissage en colonnes

# Obtenir le dictionnaire appris
D = dico_learner.components_.T  # transposer pour avoir les atomes en colonnes

# Afficher le dictionnaire appris
#print(pd.DataFrame(D))
"""

# Créer une instance de la classe ApproximateKSVD
ksvd = ApproximateKSVD(n_components=75, max_iter=500, tol=1e-8)

# Appliquer la méthode fit pour apprendre le dictionnaire
ksvd.fit(data)

# Accéder au dictionnaire appris
D = ksvd.components_
D = D.transpose()


#Implémentation OMP
def OMP(X, D, eps, N):
    m, n = D.shape                #On récupère la taille de la matrice de notre dictionnaire (m lignes, n colonnes)
    R = X                         #On vient stocker notre signal X dans la variable R - La variable R constitue le résiduel de la différence entre norm(x) et norm(D*Alpha). On initialise R avec X
    alpha = np.zeros(n)           #On initialise notre vecteur parcimonieux alpha comme étant un vecteur constitués uniquement de 0 et de taille n (Afin que le produit matriciel D*alpha soit faisable) 
    K = 0                         #On initialise une variable K à 0 - Ici K est notre compteur, afin que l'algo ne tourne pas plus de N fois 
    P = []                        #On initialise une liste P vide - Ici P constituera la liste des positions des atomes sélectionnés
    res = []                      #On initialise une liste res vide - Son unique but est de stocker les resultats du calcul qu'on effectue afin de selectionner l'atome  
    while np.linalg.norm(R,2)>eps and K<N:                    #On boucle tant que la norme 2 de R n'est pas inférieure à la précision choisie ou que l'on n'a pas dépassé le nombre max d'itérations
        for i in range(n):                                    #On boucle sur chaque colonne (=atomes)
            d = D[:,i]                                        #On sélectionne la colonne en cours et la stocke dans d
            norm_d = np.linalg.norm(d,2)                      #On calcule sa norme 
            B = abs(np.transpose(d)@R)                        #On calcule la valeur absolue du produit scalaire entre d et R 
            res.append(B/norm_d)                              #On calcule le quotien : valeur absolue du produit scalaire entre d et R / norme de la colonne en cours. Et stocke le résultat dans la liste res
        M = np.argmax(res)                                    #On vient cherche l'index de la valeur maximale de la liste res et on le stocke dans la variable M - M constitue donc la position de l'atome choisie
        P.append(M)                                           #On ajoute M a la liste des positions des atomes sélectionnés
        phi= D[:,P]                                           #Phi est une variable qui stocke tous les atomes ayant déjà été sélectionné - Phi consttitue le dictionnaire actif
        alpha[P] = np.linalg.pinv(phi)@X                      #On vient mettre à jour uniquement les coefficient de Alpha contenue dans P donc ceux dont leurs atomes correspondants ont déjà été selectionné. On vient mettre à jour ces valeurs en résolvant le problème de minimisation norm(X-D*Alpha) par rapport à Alpha
        R = X-D@alpha                                         #On vient mettre à jour notre vecteur R qui nous sert à calculer le résiduel et à stopper l'algo si la précision voulue a été atteinte
        res = []                                              #On vient vider notre liste res afin de pouvoir stocker les nouveaux calculs à la prochaine itération
        K = K+1                                               #On incrémente K
    
    return [alpha, R, K, P]                                   #On retourne 4 éléments : Alpha (vecteur parcimonieux) - R (Vecteur résiduel) - K (Nombre d'itérations de l'algorithme) - P (Liste des index des atomes sélectionnés)


#IMPLEMENTATION KSVD POUR APPRENTISSAGE DU DICTIONNAIRE
def ksvd(X, K, eps, N):
    m, n = X.shape
    R = np.zeros((N, n)) # Une matrice où je veux mettre les normes des résidus afin de voir le comportement de la méthode
    MAX_ITR = int(np.round(K/10)) # le nombre d'iteration maximal à effectuer pour l'OMP
    D0 = X[:, :K] # On initialise le dictionnaire en considérant les premières colonnes des veceturs d'apprentissage
    s = np.sqrt(np.diag(D0.T @ D0))
    for i in range(K):
        D0[:, i] = D0[:, i] / s[i] # On normalise les colonnes du dictionnaire
    Alpha = np.zeros((K, n)) # On cherche n solutions parcimonieuses pour les vecteurs d'apprentissage, les alpha_i sont de dimension égale à K
    for j in range(N):
        for i_vect in range(n):
            Alpha[:, i_vect], r = OMP(X[:, i_vect], D0, eps, MAX_ITR)[:2] #On effectue une OMP et on renvoie la solution parcimonieuse, j'ai choisi de renvoyer également les normes des résidus
            R[j, i_vect] = np.linalg.norm(r)
        D = D0
        for i_col in range(K): # Effectuer une SVD sur chaque colonne==> K colonnes issues de SVD
            idx_k = np.nonzero(Alpha[i_col, :] != 0)[0]
            if len(idx_k) > 0:
                E_k = X - D @ Alpha + np.outer(D[:, i_col], Alpha[i_col, :])
                Omega = np.zeros((n, len(idx_k)))
                for inz in range(len(idx_k)):
                    Omega[idx_k[inz], inz] = 1
                E_kR = E_k @ Omega
                U, delta, V = np.linalg.svd(E_kR)
                D0[:, i_col] = U[:, 0]
                Alpha[i_col, idx_k] = delta[0] * V[0, :]
            else:
                g = np.random.randint(0, n)
                D0[:, i_col] = X[:, g] / np.linalg.norm(X[:, g])
    return D0, Alpha, R


#D=ksvd(data.copy(),n_components,1e-3, n_iter)[0]                           #La méthode copy() permet de ne pas affecter les valeurs de la matrice initiale data

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
        for i in range(n):                                    #On boucle sur chaque colonne (=atomes)
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

#Fonction annexe afin de garder les n plus grands éléments en valeur absolu d'une liste tout en gardant 1eur position d'origine et mettre à 0 tout le reste. 
def garder_n_plus_grands_elements(liste, n):
    indices_tries = sorted(range(len(liste)), key=lambda i: abs(liste[i]), reverse=True)
    liste_tronquee = [0] * len(liste)
    for i in indices_tries[:n]:
        liste_tronquee[i] = liste[i]
    
    return np.array(liste_tronquee)


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




#IMPLEMENTATION ALGO IRLS 
def IRLS(x,D, p,kmax):
    k=0
    eps=0.1
    alphaList=[]
    alpha=D.T@np.linalg.inv((D@D.T))@x
    alphaList.append(alpha)
    while k<kmax:
        w=[]
        if k==0:
            for element in alphaList[0]:                                                                            #On itere sur chaque élément de alpha0
                w.append((abs(element)**2 + eps)**(0.5*p-1))                                                        #On calcule les w qui seront les éléments diagonaux de Q

            Q=np.diag(w)                                                                                            #On crée notre matrice diagonale Q
            alpha=Q@D.T@np.linalg.inv(D@Q@D.T)@x                                                                    #On calcule le nouveau alpha
           
            if( abs(np.linalg.norm(alpha) - np.linalg.norm(alphaList[0])) > sqrt(eps)/100):                         #Première condition si 
                k+=1

            if (abs(np.linalg.norm(alpha) - np.linalg.norm(alphaList[0])) <= sqrt(eps)/100) and eps>1e-8:           #Deuxième condition si 
                eps=eps/10                                                                                          #On modifie notre epsylon
                k+=1
            else:                                                                                                   #Troisième condition  si
                alphaList[0]=alpha
                break  
            alphaList[0]=alpha                                                                                      #On mets a jour la valeur de alpha0

        else:
            for element in alphaList[k-1]:
                w.append((abs(element)**2 + eps)**(0.5*p-1))
            
            Q=np.diag(w)                                                                                            #On crée notre matrice diagonale Q
            alpha=Q@D.T@np.linalg.inv(D@Q@D.T)@x                                                                    #On calcule le nouveau alpha
            alphaList.append(alpha)
            
            if( abs(np.linalg.norm(alpha) - np.linalg.norm(alphaList[0])) > sqrt(eps)/100):                         #Première condition si 
                k+=1
            if (abs(np.linalg.norm(alpha) - np.linalg.norm(alphaList[0])) <= sqrt(eps)/100) and eps>1e-8:           #Deuxième condition si 
                eps=eps/10                                                                                          #On modifie notre epsylon    
                k+=1
            else:                                                                                                   #Troisième condition  si
                break 
    return k,alphaList[-1]   



#x_IRLS=data.T[0]
#p_IRLS=0.5
#k_IRLS, alpha_IRLS= IRLS(x_IRLS, D, p_IRLS, 100)
#print(alpha_IRLS)


#IMPLEMENTATION DES FONCTIONS QUI CONSTRUISENT LES MATRICES DE MESURE -- (Fait en cours)
def phi1(m,n):
    return np.random.uniform(low=0.0, high=1.0, size=(m, n))

def phi2(m,n,p):
    return np.random.choice([-1, 1], size=(m,n), p=[p, 1-p])

def phi3(m,n,p):
    return np.random.choice([0, 1], size=(m,n), p=[p, 1-p])

def phi4(m,n,M):
    return np.random.normal(loc=0.0, scale=1/np.sqrt(M), size=(m, n))

#def phi5(m,n, ratio):
    #num_nonzero = int(ratio * m * n)
    #return np.random(m, n, density=ratio, data_rvs=np.random.randn).toarray()


#Implémentation du principe de compressive sensing - RECONSTRUCTION du signal

def compressedSignal(x,D,m,f):                          #x=signal, D=Dictionnaire, m=nb de valeurs dans notre vecteur compressé, f variable pour choisir matrice de mesure
    match f:
        case "phi1":
            phi=phi1(m,x.shape[0])
            return np.dot(phi, x),phi
        case "phi2": 
            phi=phi2(m,x.shape[0],0.5)
            return np.dot(phi,x),phi
        case "phi3":
            phi=phi3(m,x.shape[0],0.5)
            return np.dot(phi,x),phi
        case "phi4":
            phi=phi4(m,x.shape[0],1)
            return np.dot(phi,x), phi                   #Ici M=1, donc c'est une loi normale centrée réduite
        #case "phi5":
            #phiphi5(m,x.shape[0],0.5)
            # return np.dot(phi,x), phi                 #Ici ratio=0.5
        case _:
            return


def reconstructionSignal(y,D,phi,algo):                       #y= compressed signal, D= Dictionnaire, phi = Matrice de mesure
    match algo:
        case "OMP":                                               #1 - OMP
            A=np.dot(phi,D)
            alpha=OMP(y,A,1e-6,200)[0]                        #On lance l'algo OMP avec une précision de 10^-6 et 200 itérations max
            x=np.dot(D,alpha)                                 #On reconstuit le signal x 
            return x
        case "stOMP":                                             #2 - stOMP
            A=np.dot(phi,D)
            alpha=stOMP(y,A,2.5)[0]                           #On lance l'algo stOMP
            x=np.dot(D,alpha)                                 #On reconstuit le signal x 
            return x
        case "CoSaMP":                                            #3 - CoSaMP
            A=np.dot(phi,D)
            alpha=CoSaMP(y,A,1e-6,200,30)[0]                  #On lance l'algo CoSaMP
            x=np.dot(D,alpha)                                 #On reconstuit le signal x 
            return x   
        case "IRLS":                                            #4 - IRLS
            A=np.dot(phi,D)
            alpha=IRLS(y,A, 0.5,200)[1]                       #On lance l'algo IRLS
            x=np.dot(D,alpha)                                 #On reconstuit le signal x 
            return x
        case _:
            return

#Test fonction reconstrutionSignal
signal=vecteursTest[:,0]
y,phi=compressedSignal(signal, D, 50,"phi1")
x=reconstructionSignal(y,D,phi,"IRLS")
res=signal-x
print(res)

