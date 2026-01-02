# Importando as libs
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Lendo o arquivo
csv_data = pd.read_csv("reviews.csv")

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