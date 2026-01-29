# =========================================================
# DASHBOARD STREAMLIT – CLASSIFICATION IRIS (VERSION PROPRE)
# =========================================================

# =========================
# 1. IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# =========================
# 2. CONFIGURATION STREAMLIT
# =========================
st.set_page_config(
    page_title="Dashboard Iris – IA",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Classification des fleurs Iris avec KNN")
st.markdown("Application Streamlit pour l'analyse de données et la classification.")

# =========================
# 3. CHARGEMENT DES DONNÉES
# =========================
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["Species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df = df.drop("target", axis=1)
    return df

df = load_data()

# =========================
# 4. APERÇU DES DONNÉES
# =========================
st.subheader("📊 Aperçu du jeu de données")
st.dataframe(df.head())

st.subheader("📈 Statistiques descriptives")
st.dataframe(df.describe())

# =========================
# 5. ANALYSE UNIVARIÉE
# =========================
st.subheader("📌 Répartition des espèces")
fig, ax = plt.subplots()
sns.countplot(x="Species", data=df, ax=ax)
ax.set_title("Distribution des espèces d'Iris")
st.pyplot(fig)

# =========================
# 6. VARIABLES QUANTITATIVES
# =========================
st.subheader("📉 Histogrammes des variables quantitatives")

variables = df.columns[:-1]

for var in variables:
    fig, ax = plt.subplots()
    ax.hist(df[var], bins=20)
    ax.set_title(f"Histogramme de {var}")
    ax.set_xlabel(var)
    ax.set_ylabel("Effectif")
    st.pyplot(fig)

# =========================
# 7. ÉTUDE BIVARIÉE
# =========================
st.subheader("🔗 Relation Petal Length / Petal Width")
fig, ax = plt.subplots()
for esp in df["Species"].unique():
    subset = df[df["Species"] == esp]
    ax.scatter(subset["petal length (cm)"], subset["petal width (cm)"], label=esp)

ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.legend()
st.pyplot(fig)

# =========================
# 8. PRÉPARATION DES DONNÉES
# =========================
X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 9. MODÉLISATION KNN
# =========================
st.subheader("🤖 Modèle KNN")

k = st.slider("Choisissez le nombre de voisins (k)", 1, 10, 3)

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.success(f"🎯 Exactitude du modèle : {accuracy * 100:.2f}%")

# =========================
# 10. MATRICE DE CONFUSION
# =========================
st.subheader("📌 Matrice de confusion")

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=df["Species"].unique(),
    yticklabels=df["Species"].unique(),
    ax=ax
)
ax.set_xlabel("Prédictions")
ax.set_ylabel("Vraies classes")
st.pyplot(fig)

# =========================
# 11. RAPPORT DE CLASSIFICATION
# =========================
st.subheader("📄 Rapport de classification")
st.text(classification_report(y_test, y_pred))

# =========================
# 12. PRÉDICTION MANUELLE
# =========================
st.subheader("🧪 Tester une prédiction")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.5)

with col2:
    petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("🔮 Prédire l'espèce"):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    prediction = knn.predict(sample_scaled)
    st.info(f"🌼 Espèce prédite : **{prediction[0]}**")

# =========================
# FIN
# =========================
