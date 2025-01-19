# Importar las librerías necesarias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 2. Cargar los datos
# Cargar el dataset Iris
iris = load_iris()

# Extraer las dos primeras características (longitud y ancho del sépalo)
X = iris.data[:, :2]  # Seleccionamos las dos primeras columnas (longitud y ancho del sépalo)

# Extraer las etiquetas (especies de las flores)
y = iris.target

# 3. Dividir el conjunto de datos
# Dividir los datos en conjunto de entrenamiento (70%) y conjunto de prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Crear el modelo de clasificación
# Crear el modelo de Regresión Logística
model = LogisticRegression(max_iter=200)

# 5. Entrenar el modelo
# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# 6. Hacer predicciones
# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# 7. Calcular la precisión
# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Imprimir la precisión
print(f'Precisión del modelo: {accuracy:.4f}')

# 8. Visualización de los resultados
# Crear un DataFrame para los resultados
results_df = pd.DataFrame(X_test, columns=["Longitud del sépalo", "Ancho del sépalo"])
results_df['Especie real'] = iris.target_names[y_test]
results_df['Especie predicha'] = iris.target_names[y_pred]

# Gráfico de dispersión con las especies reales
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(data=results_df, x="Longitud del sépalo", y="Ancho del sépalo", hue="Especie real", palette="Set1", s=100)
plt.title("Especies Reales")

# Gráfico de dispersión con las especies predichas
plt.subplot(1, 2, 2)
sns.scatterplot(data=results_df, x="Longitud del sépalo", y="Ancho del sépalo", hue="Especie predicha", palette="Set1", s=100)
plt.title("Especies Predichas")

# Mostrar los gráficos
plt.tight_layout()
plt.show()