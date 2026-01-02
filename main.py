# Importando as libs
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Lendo o arquivo
csv_data = pd.read_csv('reviews.csv')

print(csv_data.head())