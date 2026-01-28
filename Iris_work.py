# Installation des l'Environment de travail à utiliser
#pip install pandas scikit-learn seaborn matplotlib 

# Importation des bibliothèques de base nécessaires 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

print("Tous les imports fonctionnent ")

df = pd.read_csv('iris.csv')

# Afficher les premières lignes du jeu de données 
print(df.head()) 
# Statistiques descriptives pour comprendre la distribution des caractéristiques 
print(df.describe()) 

# Visualisation de la répartition des classes 
sns.countplot(x='Species', data=df) 
plt.title('Distribution des espèces d\'iris') 
plt.show()

#1. Effectif de chaque modalité
effectifs = df["Species"].value_counts()
print(effectifs)

#2. Représentations graphiques des effectifs
#a. Histogramme
plt.hist(df["Species"])
plt.title("Histogramme des espèces d'Iris")
plt.xlabel("Espèces")
plt.ylabel("Effectifs")
plt.show()
#b. Secteurs
plt.pie(effectifs, labels=effectifs.index, autopct='%1.1f%%')
plt.title("Répartition des espèces d'Iris")
plt.show()
#c.Diagramme en barres groupées
plt.bar(effectifs.index, effectifs.values)
plt.title("Diagramme en barres des espèces d'Iris")
plt.xlabel("Espèces")
plt.ylabel("Effectifs")
plt.show()
#d.Diagramme en cascade
cumul = effectifs.cumsum()

plt.bar(effectifs.index, effectifs.values)
plt.plot(effectifs.index, cumul, marker='o')
plt.title("Diagramme en cascade des effectifs")
plt.xlabel("Espèces")
plt.ylabel("Effectifs cumulés")
plt.show()

#3
effectifs_Species = df["Species"].value_counts()

plt.bar(effectifs_Species.index, effectifs_Species.values)
plt.title("Effectifs des espèces d'Iris")
plt.xlabel("Espèces")
plt.ylabel("Nombre d'observations")
plt.show()

#4

#E

#Exercice 2 – Étude des variables quantitatives
#1) Résumer l’information de Petal.Length
print(df["PetalLength"].describe())




#2 Histogramme de Petal.Length
plt.hist(df["PetalLength"], bins=20)
plt.title("Histogramme de la longueur des pétales")
plt.xlabel("Longueur du pétale (cm)")
plt.ylabel("Effectif")
plt.show()



#3) Même analyse pour les autres variables quantitatives
variables = [
    "PetalWidth",
    "SepalLength",
    "SepalWidth"
]

for var in variables:
    print("\nRésumé statistique de", var)
    print(df[var].describe())
    
    plt.hist(df[var], bins=20)
    plt.title(f"Histogramme de {var}")
    plt.xlabel(var)
    plt.ylabel("Effectif")
    plt.show()

#Exercice 3 – Étude bivariée (relation entre deux variables)
#1) Nuage de points : longueur vs largeur du pétale
plt.scatter(df["PetalLength"], df["PetalWidth"])
plt.title("Relation entre la longueur et la largeur du pétale")
plt.xlabel("Longueur du pétale (cm)")
plt.ylabel("Largeur du pétale (cm)")
plt.show()



#2) Autre croisement : longueur du sépale et largeur du sépale
plt.scatter(df["SepalLength"], df["SepalWidth"])
plt.title("Relation entre la longueur et la largeur du sépale")
plt.xlabel("Longueur du sépale (cm)")
plt.ylabel("Largeur du sépale (cm)")
plt.show()

#Exercice 4 – Variable qualitative et quantitative (Boxplot)
#1) Longueur du pétale selon l’espèce
df.boxplot(column="PetalLength", by="Species")
plt.title("Longueur du pétale selon l'espèce")
plt.suptitle("")
plt.xlabel("Espèce")
plt.ylabel("Longueur du pétale (cm)")
plt.show()



#2) Autre variable : largeur du pétale selon l’espèce
df.boxplot(column="PetalWidth", by="Species")
plt.title("Largeur du pétale selon l'espèce")
plt.suptitle("")
plt.xlabel("Espèce")
plt.ylabel("Largeur du pétale (cm)")
plt.show()



#Exercice 5 – Intégration de l’espèce dans l’analyse
#1) Représentations possibles

#Nuages de points colorés selon l’espèce

#Boxplots par espèce

#Histogrammes séparés par espèce

#Matrice de corrélation

#2) Corrélations entre variables quantitatives
correlation = df.drop("Species", axis=1).corr()
print(correlation)


for esp in df["Species"].unique():
    sous_df = df[df["Species"] == esp]
    plt.scatter(
        sous_df["PetalLength"],
        sous_df["PetalWidth"],
        label=esp
    )

plt.title("Nuage de points pétales avec distinction par espèce")
plt.xlabel("Longueur du pétale (cm)")
plt.ylabel("Largeur du pétale (cm)")
plt.legend()
plt.show()

#Etape 3 : Préparer les données pour la modélisation
# Séparer les caractéristiques et la cible 
X = df.drop('Species', axis=1) 
y = df['Species'] 

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42 )

# 3. Diviser les données en ensembles d'entraînement et de test 
##from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

# Normaliser les caractéristiques 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
print("Préparation des données terminée ")

#Étape 4 : Créer et entraîner un modèle de classification (K-Nearest Neighbors) 

#1. Choisir le modèle K-Nearest Neighbors (KNN) pour la classification. 
from sklearn.neighbors import KNeighborsClassifier 
# Créer le modèle KNN 
knn = KNeighborsClassifier(n_neighbors=3) 

#2. Entraîner le modèle sur l'ensemble d'entraînement. 
# Entraîner le modèle 
knn.fit(X_train, y_train)

#Étape 5 : Évaluer le modèle
y_pred = knn.predict(X_test) 

# Afficher la matrice de confusion 
conf_matrix = confusion_matrix(y_test, y_pred) 
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
xticklabels=df['Species'].unique(), yticklabels=df['Species'].unique()) 
plt.title('Matrice de confusion') 
plt.xlabel('Prédictions') 
plt.ylabel('Vraies classes') 
plt.show() 

#3. Calculer et afficher l'exactitude, le rapport de classification et la matrice de confusion. 
# Calculer l'exactitude 
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score (y_test, y_pred) 
print(f"Exactitude du modèle : {accuracy * 100:.2f}%") 
# Afficher le rapport de classification 
print("Rapport de classification :\n", classification_report(y_test,y_pred))


# Étape 6 : Interpreter les résultats
#Étape 6 : Interprétation des résultats 
#1. Analysez les résultats du modèle, en particulier les erreurs dans la matrice de confusion:

# Réponse: En analysant la matrice de confusion, on peut identifier les classes qui sont souvent
# confondues par le modèle. Par exemple, si beaucoup d'iris Versicolor sont classés comme Virginica,
# cela indique que le modèle a des difficultés à distinguer ces deux espèces. On peut également
# observer l'exactitude globale du modèle et voir si elle est satisfaisante pour l'application envisagée.

#2. Discutez de la façon dont la normalisation des données a pu influencer les performances:

# Réponse: La normalisation des données est une étape cruciale dans le prétraitement, surtout pour
# des algorithmes comme KNN qui sont sensibles à l'échelle des caractéristiques. En normalisant
# les données, on s'assure que toutes les caractéristiques contribuent de manière égale à
# la distance calculée entre les points, ce qui peut améliorer la performance du modèle.



#Étape 7 : Optimisation du modèle et comparaison 
#1- Optimisation des hyper-paramètres 
#a- La première étape du projet consiste à choisir les hyper paramètres qui influencent les 
#performances des modèles. Pour le cas du modèle KNN, il s’agira principalement de : le 
#nombre de voisins k (1, 2, 3, 4, 5, …) et la distance (euclidienne, Manhattan, Cosine, 
#Minkowski, etc.). 

#b- La seconde étape consiste à évaluer pour chaque configuration et choisir celle offrant la 
#meilleure performance en utilisant la validation croisée ou validation simple. Deux approches 
#de recherches sont utilisées (recherche en grille et recherche aléatoire).

# Pour la recherche en grille:
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
# Entraîner la recherche en grille
grid_search.fit(X_train, y_train)
print("Meilleurs paramètres (Grid Search):", grid_search.best_params_)
best_knn_grid = grid_search.best_estimator_


# Pour la recherche aléatoire:
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_dist = {
    'n_neighbors': randint(1, 10),
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
random_search = RandomizedSearchCV(KNeighborsClassifier(), param_dist, n_iter=10, cv=5, random_state=42)
# Entraîner la recherche aléatoire
random_search.fit(X_train, y_train)
print("Meilleurs paramètres (Random Search):", random_search.best_params_)
best_knn_random = random_search.best_estimator_


#2- Entrainement des autres modèles et Comparaison  
#Essayer d'autres modèles, comme la régression logistique (Logistic Regression LR) arbres de décision 
#(Decision Tree DT), Naive Bayes (NB), Support Vector Machine (SVM), Artificial Neural Network (ANN) 
#et comparer leurs performances avec le modèle KNN. 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Créer les modèles
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC()
}
# Entraîner et évaluer chaque modèle
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_model = model.predict(X_test)
    accuracy_model = accuracy_score(y_test, y_pred_model)
    print(f"Exactitude du modèle {name} : {accuracy_model * 100:.2f}%")
# Comparer avec les meilleurs KNN
y_pred_knn_grid = best_knn_grid.predict(X_test)
accuracy_knn_grid = accuracy_score(y_test, y_pred_knn_grid)
print(f"Exactitude du meilleur KNN (Grid Search) : {accuracy_knn_grid * 100:.2f}%") 
y_pred_knn_random = best_knn_random.predict(X_test)
accuracy_knn_random = accuracy_score(y_test, y_pred_knn_random)
print(f"Exactitude du meilleur KNN (Random Search) : {accuracy_knn_random * 100:.2f}%")

#Étape 7 : Déploiement du modèle et création d’une petite interface d’API 
#1- Le déploiement du modèle sur flask se fait en utilisant les étapes suivantes :  
#a- Enregistrez le modèle en utilisant un format de sérialisation (la fonction Pickle est 
#régulièrement utilisée) pour le charger plus tard dans Flask. 
import pickle

#b- Déploiement du Modèle avec Flask : 
#Créez une application Flask qui expose une API

from flask import Flask, request, jsonify
app = Flask(__name__)
# Charger le modèle sérialisé 
#Implémentez une route /predict qui accepte une requête POST contenant les 
#caractéristiques nécessaires et renvoie les prédictions du modèle. 
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(best_knn_grid, f)
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['features']]
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})
if __name__ == '__main__':
    app.run(debug=True)