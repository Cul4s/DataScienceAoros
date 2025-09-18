# =========================================
# Dashboard Avançado - Preço de Casas
# Dataset: House Prices - Kaggle
# =========================================

# -----------------------------
# 1️⃣ Importar bibliotecas
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import streamlit as st

# -----------------------------
# 2️⃣ Carregar dados
# -----------------------------
df = pd.read_csv("train.csv")
st.title("PROJETO DATASCIENCE - Preço de Casas")

# -----------------------------
# 3️⃣ Limpeza e tratamento
# -----------------------------
df.drop_duplicates(inplace=True)
df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
df['GarageType'].fillna('None', inplace=True)
df['MasVnrType'].fillna('None', inplace=True)
df['MasVnrArea'].fillna(0, inplace=True)

# -----------------------------
# 4️⃣ Filtros interativos
# -----------------------------
st.sidebar.header("Filtros do Dashboard")
min_bath = int(df['FullBath'].min())
max_bath = int(df['FullBath'].max())
bath_filter = st.sidebar.slider("Número de Banheiros", min_bath, max_bath, (min_bath, max_bath))

min_area = int(df['GrLivArea'].min())
max_area = int(df['GrLivArea'].max())
area_filter = st.sidebar.slider("Área (sqft)", min_area, max_area, (min_area, max_area))

qual_filter = st.sidebar.multiselect("Qualidade Geral (OverallQual)", sorted(df['OverallQual'].unique()), sorted(df['OverallQual'].unique()))

df_filtered = df[
    (df['FullBath'] >= bath_filter[0]) & (df['FullBath'] <= bath_filter[1]) &
    (df['GrLivArea'] >= area_filter[0]) & (df['GrLivArea'] <= area_filter[1]) &
    (df['OverallQual'].isin(qual_filter))
]

st.subheader(f"Visualização dos Dados Filtrados ({len(df_filtered)} registros)")
st.dataframe(df_filtered.head(10))

# -----------------------------
# 5️⃣ Feature Engineering
# -----------------------------
features = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
X = df_filtered[features]
y_reg = df_filtered['SalePrice']
y_clf = (y_reg > y_reg.median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# -----------------------------
# 6️⃣ Modelos
# -----------------------------
# Regressão Linear
model_reg = LinearRegression()
model_reg.fit(X_train, y_train)
y_pred_reg = model_reg.predict(X_test)
rmse = mean_squared_error(y_test, y_pred_reg, squared=False)
st.subheader("Regressão Linear - Previsão de Preço")
st.write(f"RMSE: {rmse:.2f}")

# Classificação
model_clf = RandomForestClassifier(random_state=42)
model_clf.fit(X_train, y_clf[X_train.index])
y_pred_clf = model_clf.predict(X_test)
acc = accuracy_score(y_clf[X_test.index], y_pred_clf)
st.subheader("Classificação - Casa Cara ou Barata")
st.write(f"Accuracy: {acc:.2f}")

# -----------------------------
# 7️⃣ Gráficos interativos
# -----------------------------
st.subheader("Distribuição do Preço das Casas")
st.bar_chart(df_filtered['SalePrice'].value_counts().sort_index())

st.subheader("Área vs Preço")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=df_filtered, x='GrLivArea', y='SalePrice', hue='OverallQual', ax=ax1)
st.pyplot(fig1)

st.subheader("Preço Médio por Ano de Construção")
df_ts = df_filtered.groupby('YearBuilt')['SalePrice'].mean().reset_index()
fig2, ax2 = plt.subplots()
sns.lineplot(data=df_ts, x='YearBuilt', y='SalePrice', ax=ax2)
st.pyplot(fig2)

st.subheader("Preço Médio por Número de Banheiros")
bath_avg = df_filtered.groupby('FullBath')['SalePrice'].mean()
st.bar_chart(bath_avg)

# -----------------------------
# 8️⃣ Previsão em tempo real
# -----------------------------
st.subheader("Prever Preço de Casa em Tempo Real")
st.write("Selecione os valores da casa:")

live_overall = st.slider("Qualidade Geral (OverallQual)", int(df['OverallQual'].min()), int(df['OverallQual'].max()), 5)
live_area = st.number_input("Área (sqft)", min_value=int(df['GrLivArea'].min()), max_value=int(df['GrLivArea'].max()), value=1500)
live_garage = st.slider("Número de Garagens", int(df['GarageCars'].min()), int(df['GarageCars'].max()), 1)
live_bsmt = st.number_input("Área do Porão (sqft)", min_value=0, max_value=int(df['TotalBsmtSF'].max()), value=800)
live_bath = st.slider("Número de Banheiros", int(df['FullBath'].min()), int(df['FullBath'].max()), 2)
live_year = st.slider("Ano de Construção", int(df['YearBuilt'].min()), int(df['YearBuilt'].max()), 2000)

# Criar dataframe para previsão
live_data = pd.DataFrame([[live_overall, live_area, live_garage, live_bsmt, live_bath, live_year]], columns=features)
predicted_price = model_reg.predict(live_data)[0]
st.success(f"💰 Preço Previsto: ${predicted_price:,.2f}")
