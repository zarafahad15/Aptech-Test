# app.py — Complete Streamlit Titanic Classifier App

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(page_title="Titanic Classifier", layout="wide")

# --------------------- TITLE SECTION -------------------
app_title = st.text_input("App Title", value="🚢 Titanic Classifier — Interactive ML App")
st.title(app_title)
sub_heading = st.text_input("Page heading", value="Train and compare ML models on the Titanic dataset")
st.markdown(f"#### {sub_heading}")

# --------------------- LOAD DATA -----------------------
@st.cache_data
def load_data():
    return sns.load_dataset("titanic")

df = load_data()

st.subheader("📊 Titanic Dataset Preview")
st.dataframe(df.head())
st.write("✅ Dataset shape:", df.shape)

# --------------------- SIDEBAR SETTINGS ----------------
st.sidebar.header("⚙️ Model Controls")

alg = st.sidebar.selectbox(
    "Choose Algorithm",
    ["Random Forest", "Decision Tree", "SVC", "KNN (KNeighborsClassifier)"]
)

st.sidebar.subheader("🔧 Train/Test Split")
test_size = st.sidebar.slider("Test size fraction", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

# --------------------- FEATURE ENGINEERING ------------
FEATURES = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
selected_features = st.multiselect("Select features", FEATURES, default=FEATURES)

@st.cache_data
def preprocess(df, features):
    df2 = df[features + ['survived']].copy()

    # Fill numeric columns
    num_cols = df2.select_dtypes(include=['number']).columns.tolist()
    for col in num_cols:
        df2[col] = df2[col].fillna(df2[col].median())

    # Fill categorical columns
    cat_cols = df2.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        df2[col] = df2[col].astype(str).fillna('missing')

    # One-hot encode
    X = pd.get_dummies(df2.drop('survived', axis=1), drop_first=True)
    y = df2['survived'].astype(int)
    return X, y

X, y = preprocess(df, selected_features)

st.subheader("⚙️ Processed Features (First 5 rows)")
st.dataframe(X.head())

# --------------------- TRAIN/TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

# Optional scaling
numeric_cols = X_train.select_dtypes(include=['float', 'int']).columns.tolist()
if st.sidebar.checkbox("Scale numeric features (recommended for SVC/KNN)", value=True):
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

st.write("✅ Train shape:", X_train.shape, " | Test shape:", X_test.shape)

# --------------------- HYPERPARAMETERS -----------------
st.sidebar.subheader("⚙️ Hyperparameters")

model = None
if alg == "Random Forest":
    n_estimators = st.sidebar.number_input("n_estimators", 10, 1000, 100, 10)
    max_depth = st.sidebar.number_input("max_depth (0=None)", 0, 100, 0)
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=None if max_depth == 0 else int(max_depth),
        random_state=int(random_state)
    )

elif alg == "Decision Tree":
    max_depth = st.sidebar.number_input("max_depth (0=None)", 0, 100, 0)
    criterion = st.sidebar.selectbox("criterion", ["gini", "entropy"])
    model = DecisionTreeClassifier(
        max_depth=None if max_depth == 0 else int(max_depth),
        criterion=criterion,
        random_state=int(random_state)
    )

elif alg == "SVC":
    C = st.sidebar.number_input("C (regularization)", 0.01, 10.0, 1.0, 0.01)
    kernel = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly"])
    gamma = st.sidebar.selectbox("gamma", ["scale", "auto"])
    model = SVC(C=float(C), kernel=kernel, gamma=gamma, probability=True, random_state=int(random_state))

elif alg == "KNN (KNeighborsClassifier)":
    n_neighbors = st.sidebar.number_input("n_neighbors", 1, 50, 5)
    weights = st.sidebar.selectbox("weights", ["uniform", "distance"])
    p = st.sidebar.selectbox("p (1=manhattan, 2=euclidean)", [1, 2], index=1)
    model = KNeighborsClassifier(n_neighbors=int(n_neighbors), weights=weights, p=int(p))

# --------------------- TRAINING & METRICS --------------
train_button = st.sidebar.button("🚀 Train Model")

if train_button:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    st.subheader("📈 Evaluation Metrics")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**Precision:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# --------------------- EDA SECTION ---------------------
st.subheader("📊 Exploratory Data Analysis (EDA)")

if st.checkbox("Show survival counts"):
    fig, ax = plt.subplots()
    sns.countplot(x='survived', data=df, ax=ax)
    ax.set_xticklabels(['Not Survived', 'Survived'])
    st.pyplot(fig)

if st.checkbox("Show age distribution"):
    fig, ax = plt.subplots()
    sns.histplot(df['age'].dropna(), kde=True, bins=30, ax=ax)
    st.pyplot(fig)

if st.checkbox("Show fare boxplot"):
    fig, ax = plt.subplots()
    sns.boxplot(x='survived', y='fare', data=df, ax=ax)
    st.pyplot(fig)

if st.checkbox("Survival by sex"):
    fig, ax = plt.subplots()
    sns.countplot(x='sex', hue='survived', data=df, ax=ax)
    st.pyplot(fig)
