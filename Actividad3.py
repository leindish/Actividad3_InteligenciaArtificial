# Importar las librerías necesarias
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np   # Para operaciones numéricas
from sklearn.model_selection import train_test_split  # Para dividir el dataset
from sklearn.ensemble import RandomForestRegressor  # Para el modelo de Random Forest
from sklearn.metrics import mean_squared_error  # Para calcular el error
from sklearn.model_selection import GridSearchCV  # Para ajuste de hiperparámetros

# Definición de las variables
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
horas_dia = ['Mañana', 'Tarde', 'Noche']
condiciones_clima = ['Soleado', 'Nublado', 'Lluvioso']
capacidad_vehiculo = [50, 60, 80, 100]

# Fijar la semilla para reproducibilidad
np.random.seed(42)

# Función para generar el tiempo de espera basado en condiciones de tráfico y demanda
def generar_tiempo_espera(trafico, demanda):
    return max(5, np.random.normal(15 + 0.5 * trafico + 0.1 * demanda, 3))

# Función para generar tráfico en función del día y la hora
def generar_trafico(dia, hora, clima):
    base_trafico = 3 if dia in ['Sábado', 'Domingo'] else 7  # Menos tráfico los fines de semana
    base_trafico += 3 if hora == 'Mañana' else (2 if hora == 'Tarde' else 1)  # Más tráfico en la mañana
    if clima == 'Lluvioso':
        base_trafico += 2  # Aumentar tráfico si está lloviendo
    return min(10, max(0, np.random.normal(base_trafico, 2)))

# Función para generar demanda de pasajeros
def generar_demanda(hora, capacidad):
    base_demanda = 70 if hora == 'Mañana' else (50 if hora == 'Tarde' else 30)
    return min(capacidad, max(0, np.random.normal(base_demanda, 15)))

# Generar un dataset con 500 muestras
datos = []
for _ in range(500):
    dia = np.random.choice(dias_semana)
    hora = np.random.choice(horas_dia)
    clima = np.random.choice(condiciones_clima)
    capacidad = np.random.choice(capacidad_vehiculo)
    trafico = generar_trafico(dia, hora, clima)
    demanda = generar_demanda(hora, capacidad)
    tiempo_espera = generar_tiempo_espera(trafico, demanda)

    datos.append([dia, hora, clima, tiempo_espera, trafico, capacidad, demanda])

# Convertir la lista de datos a un DataFrame de pandas
df = pd.DataFrame(datos, columns=['Día de la Semana', 'Hora del Día', 'Condición Climática', 
                                  'Tiempo de Espera (minutos)', 'Condiciones de Tráfico (escala 0-10)', 
                                  'Capacidad del Vehículo', 'Demanda de Pasajeros'])

# Mostrar las primeras filas del dataset
print(f"\nPrimeras Filas del Dataset\n")
print(df.head())
print(f"\n")
# Convertir variables categóricas a numéricas (OneHotEncoding)
df_encoded = pd.get_dummies(df, columns=['Día de la Semana', 'Hora del Día', 'Condición Climática'])

# Separar características (X) y etiqueta (y)
X = df_encoded.drop('Tiempo de Espera (minutos)', axis=1)
y = df_encoded['Tiempo de Espera (minutos)']

# Dividir el dataset en conjunto de entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo de Random Forest
model_rf = RandomForestRegressor(random_state=42)

# Definir los hiperparámetros para ajuste
param_grid = {
    'n_estimators': [100, 200, 300], # Número de árboles en el bosque
    'max_depth': [10, 20, 30], # Profundidad máxima de los árboles
    'min_samples_split': [2, 5, 10],  # Muestras mínimas necesarias para dividir un nodo
    'min_samples_leaf': [1, 2, 4] # Muestras mínimas necesarias para formar una hoja
}

# Realizar la búsqueda de hiperparámetros con validación cruzada
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Ajustar el modelo con los datos de entrenamiento
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_model = grid_search.best_estimator_

# Predecir y calcular el error en el conjunto de prueba
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Imprimir los resultados
print(f"\nMejor modelo: {grid_search.best_params_}\n")
print(f"Error cuadrático medio (RMSE): {rmse}\n")
