import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Asegúrate de tener instalada la librería imblearn
# Puedes instalarla usando: pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

# Cargar los datos
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Verificar valores faltantes
print(data.isnull().sum())

# Limpiar los datos
data = data.dropna(subset=['bmi'])  # Eliminar filas con valores faltantes en 'bmi'
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())  # Rellenar valores faltantes en 'bmi' con la media

# Convertir columnas categóricas a variables numéricas
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
data['ever_married'] = data['ever_married'].map({'No': 0, 'Yes': 1})
data['work_type'] = data['work_type'].astype('category').cat.codes
data['Residence_type'] = data['Residence_type'].map({'Rural': 0, 'Urban': 1})
data['smoking_status'] = data['smoking_status'].astype('category').cat.codes

# Separar características y variable objetivo
X = data.drop('stroke', axis=1)
y = data['stroke']

# Balancear las clases
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Escalar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajuste de hiperparámetros para Random Forest
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='r2')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

# Ajuste de hiperparámetros para XGBoost
xgb = XGBRegressor(random_state=42)
param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='r2')
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_

# Redes Neuronales
nn = MLPRegressor(random_state=42, max_iter=500)
param_grid_nn = {
    'hidden_layer_sizes': [(50,), (100,)],
    'learning_rate_init': [0.001, 0.01]
}
grid_nn = GridSearchCV(nn, param_grid_nn, cv=3, scoring='r2')
grid_nn.fit(X_train, y_train)
best_nn = grid_nn.best_estimator_

# Regresión Lineal
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

# Random Forest
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# XGBoost
best_xgb.fit(X_train, y_train)
y_pred_xgb = best_xgb.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# Redes Neuronales
best_nn.fit(X_train, y_train)
y_pred_nn = best_nn.predict(X_test)
r2_nn = r2_score(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)

# Imprimir resultados
print("Regresión Lineal:")
print(f"R²: {r2_lr}")
print(f"MSE: {mse_lr}")
print(f"MAE: {mae_lr}")

print("\nRandom Forest:")
print(f"R²: {r2_rf}")
print(f"MSE: {mse_rf}")
print(f"MAE: {mae_rf}")

print("\nXGBoost:")
print(f"R²: {r2_xgb}")
print(f"MSE: {mse_xgb}")
print(f"MAE: {mae_xgb}")

print("\nRedes Neuronales:")
print(f"R²: {r2_nn}")
print(f"MSE: {mse_nn}")
print(f"MAE: {mae_nn}")

# Crear el gráfico de correlación
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()

# Visualizar resultados de los modelos
modelos = ['Regresión Lineal', 'Random Forest', 'XGBoost', 'Redes Neuronales']
r2_scores = [r2_lr, r2_rf, r2_xgb, r2_nn]
mse_scores = [mse_lr, mse_rf, mse_xgb, mse_nn]
mae_scores = [mae_lr, mae_rf, mae_xgb, mae_nn]

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
sns.barplot(x=modelos, y=r2_scores)
plt.title('R² Scores')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
sns.barplot(x=modelos, y=mse_scores)
plt.title('MSE Scores')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
sns.barplot(x=modelos, y=mae_scores)
plt.title('MAE Scores')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
