# app.py
Heart Disease Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score
)
st.set_page_config(
    page_title="Dashboard Heart Disease",
    page_icon="❤️",
    layout="wide"
)
df = pd.read_csv("heart_disease.csv")
st.sidebar.title("❤️ Dashboard")

menu = st.sidebar.radio(
    "Menu",
    [
        "Home",
        "Dataset",
        "Data Cleaning",
        "Statistik Deskriptif",
        "Visualisasi",
        "Logistic Regression",
        "Evaluasi Model",
        "Prediksi"
    ]
)
if menu == "Home":

    st.title("❤️ Dashboard Heart Disease")

    st.markdown("""
    Dashboard Analisis Regresi Logistik
    untuk Prediksi Penyakit Jantung.

    Dataset:
    - 1025 Observasi
    - 14 Variabel

    Metode:
    - Logistic Regression
    - Confusion Matrix
    - ROC Curve
    - Odds Ratio
    """)
elif menu == "Dataset":

    st.header("Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Baris", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])
    col3.metric("Duplicate", df.duplicated().sum())

    st.dataframe(df)
elif menu == "Data Cleaning":

    st.header("Missing Value")

    st.dataframe(df.isnull().sum())

    st.header("Duplicate")

    st.write("Jumlah Duplicate:", df.duplicated().sum())

    if st.checkbox("Tampilkan Duplicate"):

        st.dataframe(df[df.duplicated(keep=False)])
elif menu == "Statistik Deskriptif":

    st.header("Statistik Deskriptif")

    st.dataframe(df.describe())
elif menu == "Visualisasi":

    st.header("📊 Visualisasi Variabel")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    pilihan = st.selectbox(
        "Pilih Variabel",
        numeric_cols
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(df[pilihan], bins=20)

    ax.set_title(f"Histogram {pilihan}")

    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 1.8))

    sns.boxplot(x=df[pilihan], ax=ax2)

    ax2.set_title(f"Boxplot {pilihan}")

    st.pyplot(fig2)
elif menu == "Logistic Regression":

    st.header("Model Logistic Regression")

    X = df.drop("target", axis=1)
    y = df["target"]

    model = LogisticRegression(max_iter=1000)

    model.fit(X, y)

    coef = pd.DataFrame({
        "Variable": X.columns,
        "Coefficient": model.coef_[0],
        "Odds Ratio": np.exp(model.coef_[0])
    })

    st.dataframe(coef)
elif menu == "Evaluasi Model":

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=123
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    sens = recall_score(y_test, pred)

    TN, FP, FN, TP = cm.ravel()

    spec = TN / (TN + FP)

    col1, col2 = st.columns(2)

    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")

    col1.metric("Sensitivity", f"{sens:.3f}")
    col2.metric("Specificity", f"{spec:.3f}")

    prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, prob)

    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr)

    ax.plot([0, 1], [0, 1], "--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    st.pyplot(fig)

    st.success(f"AUC : {roc_auc:.3f}")

elif menu == "Prediksi":

    st.header("Prediksi Penyakit Jantung")

    age = st.number_input("Age", 20,100,50)

    sex = st.number_input("Sex",0,1,1)

    cp = st.number_input("CP",0,3,0)

    trestbps = st.number_input("Trestbps",50,250,120)

    chol = st.number_input("Chol",100,600,200)

    fbs = st.number_input("FBS",0,1,0)

    restecg = st.number_input("RestECG",0,2,1)

    thalach = st.number_input("Thalach",50,250,150)

    exang = st.number_input("Exang",0,1,0)

    oldpeak = st.number_input("Oldpeak",0.0,10.0,1.0)

    slope = st.number_input("Slope",0,2,1)

    ca = st.number_input("CA",0,4,0)

    thal = st.number_input("Thal",0,3,2)
