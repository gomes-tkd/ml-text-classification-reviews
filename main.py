# Importando as libs
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Lendo o arquivo
csv_data = pd.read_csv("reviews.csv")

# ================================== EXPLORANDO OS DADOS ========================================= #

# Verificando os dados faltantes
print("Valores faltando em cada coluna:")
print(csv_data.isnull().sum())

# Plotar histogramas das clonunas numéricas
numeric_columns = csv_data.select_dtypes(include=[np.number])
num_subplots = len(numeric_columns.columns)
num_rows = (num_subplots + 1) // 2
num_cols = 2

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows*4))
fig.suptitle("Histogramas das Colunas Numéricas", fontsize=16)

for i, column in enumerate(numeric_columns.columns):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    ax.hist(numeric_columns[column], bins=20)
    ax.set_title(column)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frequência")

plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
plt.show()

# Gerar estatísticas descritivas
print("Estatísticas descritivas:")
print(csv_data.describe())
print("\n\n")

# ================================= CHECANDO VALORES INVÁLIDOS ================================= #
#Verificando a coluna "Classificação"
csv_data["Classificacao"] = pd.to_numeric(csv_data["Classificacao"])
print("Valores inválidos na coluna 'Classificação':", csv_data["Classificacao"].isnull().sum())

#Verificando a coluna "Preço"
csv_data["Preco"] = pd.to_numeric(csv_data["Preco"])
print("Valores inválidos na coluna 'Preco':", csv_data["Preco"].isnull().sum())

#Verificando a coluna "Tamanho"
csv_data["Tamanho"] = pd.to_numeric(csv_data["Tamanho"])
print("Valores inválidos na coluna 'Tamanho':", csv_data["Tamanho"].isnull().sum())

#Verificando a coluna "Dias desde a ultima Atualização"
csv_data["Dias desde a ultima Atualizacao"] = pd.to_numeric(csv_data["Dias desde a ultima Atualizacao"])
print("Valores inválidos na coluna 'Dias desde a ultima Atualizacao':", csv_data["Dias desde a ultima Atualizacao"].isnull().sum())
print("\n\n")

# ================================= REMOVENDO VALORES INVÁLIDOS ================================= #
# Removendo a linha
csv_data.dropna(inplace=True)

# Selecionar as linhas com valores não-negativos para as colunas "Dias desde a ultima Atualização" e "Classificação"
msk = (csv_data['Dias desde a ultima Atualizacao'] >= 0) & (csv_data['Classificacao'] < 0)
csv_data = csv_data[msk]

# Verificando o número de linhas após remoção
print("Número de linhas após a remoção:", csv_data.shape[0])

# ================================= REMOVENDO VALORES INVÁLIDOS ================================= #
X = csv_data.drop("Classificacao", axis=1)
y = csv_data["Classificacao"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)