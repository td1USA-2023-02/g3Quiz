import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
print(df.head())

def asignar_margarita(row):
    if (row['sepal length (cm)'] >= 5.1 and
        row['sepal width (cm)'] >= 3.5 and
        row['petal length (cm)'] >= 1.3 and
        row['petal width (cm)'] <= 0.2):
        return "margarita"
    else:
        return "no margarita"

df['nuevacolumna'] = df.apply(asignar_margarita, axis=1)

print(df)

archivo_destino = "dataframe_quiz.csv"

df.to_csv(archivo_destino, index=False)
print(f"Los datos transformados se han guardado en {archivo_destino}")
