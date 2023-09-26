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

#Gráfico histograma de acuerdo a longitud de sépalo
sns.histplot(data=df, x="sepal length (cm)", hue="nuevacolumna", bins=20)
plt.title("Histograma de Longitud del Sépalo")
plt.xlabel("Longitud del Sépalo (cm)")
plt.ylabel("Frecuencia")
plt.show()
#Gráfico de violin de acuerdo a longitud de sepalo
sns.violinplot(x="nuevacolumna", y="sepal length (cm)", data=df)
plt.title("Diagrama de Violín: Largo del Sépalo por Especie")
plt.show()

#Conclusiones según gráficos de histograma y de violin, ambos con longitud del sepalo
#Para el histograma este nos indica la frecuencia que hay de aquellas que son margaritas y aquellas que no
#la gráfica nos dice que únicamente se encuentran 8 margaritas de toda la data disponible y estas se encuentran en un rango de sépalo de 5 a 5,5 centímetros
#fuera de esos límites, no se encuentra ninguna margarita, para el caso de la longitud del sépalo

#Para el diagrama de violin 