# Importando as libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Lendo o arquivo
csv_data = pd.read_csv("reviews.csv")

# ================================== EXPLORANDO OS DADOS ========================================= #

# Verificando os dados faltantes
print("Valores faltando em cada coluna:")
print(csv_data.isnull().sum())

# Plotar histogramas das colunas numéricas
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

plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
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

#Verificando a coluna "Dias desde a última Atualização"
csv_data["Dias desde a ultima Atualizacao"] = pd.to_numeric(csv_data["Dias desde a ultima Atualizacao"])
print("Valores inválidos na coluna 'Dias desde a ultima Atualizacao':", csv_data["Dias desde a ultima Atualizacao"].isnull().sum())
print("\n\n")

# ================================= REMOVENDO VALORES INVÁLIDOS ================================= #
# Removendo a linha
csv_data.dropna(inplace=True)

# Selecionar as linhas com valores não-negativos para as colunas "Dias desde a última Atualização" e "Classificação"
msk = (csv_data['Dias desde a ultima Atualizacao'] >= 0) & (csv_data['Classificacao'] >= 0)
csv_data = csv_data[msk]

# Verificando o número de linhas após remoção
print("Número de linhas após a remoção:", csv_data.shape[0])

# ================================= REMOVENDO VALORES INVÁLIDOS ================================= #
X = csv_data.drop("Classificacao", axis=1)
y = csv_data["Classificacao"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ================================= CORRELAÇÃO ENTRE AS FEATURES ================================ #
# Removendo a coluna "Categoria" antes de calcular a matriz de correlação
X_train_numeric = X_train.drop("Categoria", axis=1)

# Converter as colunas relevantes para tipos numéricos
numeric_columns = ["No de Reviews", "No de Instalacoes", "Tamanho", "Preco", "Dias desde a ultima Atualizacao"]
X_train_numeric[numeric_columns] = X_train_numeric[numeric_columns].apply(pd.to_numeric, errors="coerce")

# Calcular a matriz de correlação
correlation_matrix = X_train_numeric.corr()
print("Matriz de Correlação")
print(correlation_matrix)
print("\n\n")

# ================================= CRIANDO O GRÁFICO DE DISPERSÃO ================================ #
#
columns_to_plot = ["No de Reviews", "No de Instalacoes", "Tamanho", "Preco", "Dias desde a ultima Atualizacao"]
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
fig.tight_layout(pad=5.0)

for i, column in enumerate(columns_to_plot):
    row = i // 3
    col = i % 3
    axes[row, col].scatter(X_train_numeric[column], y_train)
    axes[row, col].set_title(f'Classificação vs {column}')
    axes[row, col].set_xlabel(column)
    axes[row, col].set_ylabel("Classificação")

plt.show()
print("\n\n")

# ================================= PRÉ-PROCESSAMENTO DE DADOS ================================= #
# Declarar as variáveis num_col e cat_col
num_col = ["No de Reviews", "No de Instalacoes", "Tamanho", "Preco", "Dias desde a ultima Atualizacao"]
cat_col = ["Categoria"]

# Substituir os valores ausentes na coluna "Tamanho
imp = SimpleImputer(strategy="mean")
tf_num = imp.fit_transform(X_train[num_col])

# Escalando as colunas numéricas
scaler = StandardScaler()
tf_num = scaler.fit_transform(tf_num)

# Codificar a coluna "Categoria" usando a codificação one-hot
ohe = OneHotEncoder(sparse_output=False, drop="first")
tf_cat = ohe.fit_transform(X_train[cat_col])

# Conectar os arrays tf_num e tf_cat ao longo do eixo 1
X_train_transformed = np.concatenate((tf_num, tf_cat), axis=1)

# Verificar o resultado imprimindo o primeiro exemplo
print("Exemplo transformado:")
print(X_train_transformed)
print("\n\n")

# ================================= TREINANDO O MODELO ========================================= #
# Instanciando o objeto LinearRegression
model = LinearRegression()

# Treinando o modelo usando os dados de treinamento transformados
model.fit(X_train_transformed, y_train)

# Imprimindo os coeficientes (coef_) e o intercepto (intercept_)
print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)
print("\n\n")

# ================================= AVALIANDO O MODELO ========================================= #
# Avaliação no conjunto de dados de treinamento
y_pred_train = model.predict(X_train_transformed)
rmse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

print("RMSE train:", rmse_train)
print("R2 train:", r2_train)
print("\n")
# Transformação dos dados de teste
tf_num_test = imp.transform(X_test[num_col])
tf_num_test = scaler.transform(tf_num_test)
tf_cat_test = ohe.transform(X_test[cat_col])
X_test_transformed = np.concatenate((tf_num_test, tf_cat_test), axis=1)

# Avaliação no conjunto de dados de teste
y_pred_test = model.predict(X_test_transformed)
rmse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("RMSE test:", rmse_test)
print("R2 test:", r2_test)
print("\n\n")

# ======================= USANDO COLUMN TRANSFORMER E PIPELINE ================================== #
# Criação do pipeline de pré-processamento para as colunas numéricas
num_preprocessing = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
])

# Criação do ColumnTransformer
full_processing = ColumnTransformer([
    ("num", num_preprocessing, num_col),
    ("cat", OneHotEncoder(sparse_output=False, drop="first"), cat_col),
])

# Criação do pipeline final
final_pipeline = Pipeline([
    ("preprocessing", full_processing),
    ("model", LinearRegression()),
])
print("\n\n")

# ======================= USANDO PIPELINE PARA TREINAR O MODELO ================================= #
# Treinando o modelo usando o pipeline final
final_pipeline.fit(X_train, y_train)

# ======================= AVALIANDO O MODELO COM PIPELINE ======================================= #
# Avaliação do modelo no conjunto de treinamento
y_pred_train = final_pipeline.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)

print("Avaliação no conjunto de treinamento:")
print("RMSE train:", rmse_train)
print("R2 train:", r2_train)
print("\n")
# Avaliação do modelo no conjunto de teste
y_pred_test = final_pipeline.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print("Avaliação no conjunto de teste:")
print("RMSE test:", rmse_test)
print("R2 test:", r2_test)
print("\n\n")

# ======================= VALIDAÇÃO CRUZADA USANDO PIPELINE ===================================== #
# Validação Cruzada com RMSE negativo como critério de pontuação
rmse_scores = cross_val_score(final_pipeline, X_train, y_train, cv=4, scoring="neg_mean_squared_error")
rmse_scores_abs = np.abs(rmse_scores)

print("Validação cruzada com RMSE:")
print("Scores:", rmse_scores_abs)
print("Média:", np.mean(rmse_scores_abs))
print("\n")

# Validação Cruzada com R2 como critério de pontuação
r2_scores = cross_val_score(final_pipeline, X_train, y_train, cv=4, scoring="r2")
print("Validação cruzada com R2:")
print("Scores:", r2_scores)
print("Média:", np.mean(r2_scores))
print("\n\n")