# Predicción de Riesgo de Stroke con Modelos de Machine Learning

Este proyecto utiliza técnicas de machine learning para predecir el riesgo de un accidente cerebrovascular (stroke) en función de varias características de salud. El objetivo es explorar y evaluar diferentes modelos de machine learning para identificar cuál es el más eficaz para predecir esta condición.

## Tecnologías y Bibliotecas Utilizadas

- **Pandas**: Para la manipulación y análisis de datos.
- **Seaborn**: Para la visualización de datos.
- **Matplotlib**: Para la creación de gráficos.
- **Scikit-learn**: Para la implementación de modelos de machine learning y evaluación de su desempeño.
- **XGBoost**: Para la implementación del modelo XGBoost.
- **imblearn**: Para balancear las clases con SMOTE (Synthetic Minority Over-sampling Technique).
- **MLPRegressor**: Para la implementación de redes neuronales.

## Descripción del Proyecto

El dataset utilizado en este proyecto es **healthcare-dataset-stroke-data.csv**, que contiene información sobre pacientes, incluidas características como edad, género, índice de masa corporal (BMI), tipo de trabajo, y si son fumadores o no. El objetivo es predecir si un paciente tendrá un accidente cerebrovascular basado en estos atributos.

El proyecto implementa los siguientes modelos para la predicción:

1. **Regresión Lineal**
2. **Random Forest**
3. **XGBoost**
4. **Redes Neuronales (MLPRegressor)**

El flujo del proyecto es el siguiente:

1. **Carga y limpieza de datos**: El dataset se carga, se manejan los valores faltantes y se convierten las variables categóricas en numéricas.
2. **Balanceo de clases**: Se utiliza SMOTE para balancear las clases debido a la desproporción entre casos de accidente cerebrovascular y no.
3. **Escalado de características**: Se estandarizan las características para que todos los atributos tengan la misma escala.
4. **División de los datos**: Los datos se dividen en conjuntos de entrenamiento y prueba.
5. **Entrenamiento y evaluación de modelos**: Los modelos se entrenan y se evalúan utilizando métricas como R², MSE (Error cuadrático medio) y MAE (Error absoluto medio).
6. **Visualización de resultados**: Se visualizan las métricas de desempeño de cada modelo y se crea una matriz de correlación.

## Cómo Ejecutar el Proyecto

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu-usuario/nombre-del-repositorio.git
   cd nombre-del-repositorio
