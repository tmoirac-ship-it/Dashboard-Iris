# ===============================
# IMPORTS
# ===============================
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# CONFIG STREAMLIT
# ===============================
st.set_page_config(page_title="Iris Dashboard", layout="wide")
st.title("🌸 Analyse et classification du jeu de données Iris")

# ===============================
# CHARGEMENT DES DONNÉES
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("iris.csv")

df = load_data()
st.subheader("Aperçu du jeu de données")
st.dataframe(df.head())

# ===============================
# STATISTIQUES DESCRIPTIVES
# ===============================
st.subheader("Statistiques descriptives")
st.write(df.describe())

# ===============================
# DISTRIBUTION DES CLASSES
# ===============================
st.subheader("Distribution des espèces")

fig, ax = plt.subplots()
sns.countplot(x="Species", data=df, ax=ax)
st.pyplot(fig)

# ===============================
# VISUALISATION PETAL
# ===============================
st.subheader("Relation Petal Length / Petal Width")

fig, ax = plt.subplots()
sns.scatterplot(
    data=df,
    x="PetalLength",
    y="PetalWidth",
    hue="Species",
    ax=ax
)
st.pyplot(fig)

# ===============================
# PRÉPARATION DES DONNÉES
# ===============================
X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# MODÈLE KNN
# ===============================
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# ===============================
# ÉVALUATION
# ===============================
st.subheader("Évaluation du modèle")

accuracy = accuracy_score(y_test, y_pred)
st.success(f"Exactitude du modèle : {accuracy * 100:.2f}%")

st.text("Rapport de classification")
st.text(classification_report(y_test, y_pred))

# ===============================
# MATRICE DE CONFUSION
# ===============================
st.subheader("Matrice de confusion")

conf_matrix = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(
    conf_matrix,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=df["Species"].unique(),
    yticklabels=df["Species"].unique(),
    ax=ax
)
ax.set_xlabel("Prédictions")
ax.set_ylabel("Vraies classes")
st.pyplot(fig)

# ===============================
# PRÉDICTION UTILISATEUR
# ===============================
st.subheader("Prédiction manuelle")

sepal_length = st.number_input("Sepal Length", 0.0)
sepal_width = st.number_input("Sepal Width", 0.0)
petal_length = st.number_input("Petal Length", 0.0)
petal_width = st.number_input("Petal Width", 0.0)

if st.button("Prédire l'espèce"):
    input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn.predict(input_data)
    st.info(f"🌼 Espèce prédite : **{prediction[0]}**")
