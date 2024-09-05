
===========================================================
PREDICCIÓN DE PROPENSIÓN A LA COMPRA DE PLANES DE PENSIONES
===========================================================

Descripción del Proyecto
------------------------
Este proyecto tiene como objetivo predecir la propensión de los clientes a adquirir el servicio "Planes de Pensiones". 
El análisis se basa en datos históricos de ventas de una empresa para enfocar las próximas acciones de marketing y optimizar los recursos.

Análisis Descriptivo
--------------------
Primero, realicé un análisis descriptivo de los datos utilizando Power BI. Creé un dashboard para visualizar indicadores clave (KPIs) 
y realizar un análisis evolutivo de la empresa. Este análisis ayudó a identificar que el producto "Planes de Pensiones" 
era el más rentable con el menor esfuerzo, por lo que se decidí enfocar las campañas de marketing en este producto.

Análisis Predictivo
-------------------
Para el análisis predictivo, utilicé un enfoque de Machine Learning para identificar los clientes potenciales 
que podrían estar interesados en adquirir "Planes de Pensiones". El proceso se realizó en un notebook de Google Colab 
con conexión a Google Drive para acceder a los datos almacenados en archivos CSV.

Librerías y Herramientas Utilizadas
-----------------------------------
- Gestión de Datos:
  * pandas y numpy: Para la manipulación y análisis de datos.
  
- Visualización de Datos:
  * seaborn y matplotlib: Para la creación de gráficos y visualizaciones.

- Preprocesamiento de Datos:
  * sklearn.preprocessing.StandardScaler: Para escalar y normalizar las características.

- Modelado y Evaluación:
  * Modelos de Clasificación:
    - sklearn.tree.DecisionTreeClassifier: Árboles de decisión.
    - sklearn.ensemble.RandomForestClassifier: Ensamble de árboles.
    - sklearn.ensemble.GradientBoostingClassifier: Boosting de árboles.
    - sklearn.neighbors.KNeighborsClassifier: Vecinos más cercanos.
    - sklearn.naive_bayes.GaussianNB: Clasificación Naive Bayes.
    - xgboost.XGBClassifier: Implementación de Gradient Boosting eficiente.
    
  * Métricas de Evaluación:
    - accuracy_score: Precisión del modelo.
    - roc_auc_score: Área bajo la curva ROC.
    - roc_curve: Curva ROC.
    - confusion_matrix: Matriz de confusión.

  * Evaluación Cruzada:
    - cross_val_score: Validación cruzada para evaluar el rendimiento.

Selección del Modelo
--------------------
Se compararon seis modelos de clasificación para identificar el más adecuado. Se iteraron diferentes técnicas de limpieza y 
preprocesamiento de datos. Finalmente, se eligió el *Gradient Boosting Classifier* debido a su equilibrio entre la precisión en entrenamiento y prueba.

Aplicación Web
--------------
Desarrollé una aplicación web utilizando Streamlit, que permite a los usuarios probar el modelo con distintos 
perfiles de clientes y visualizar los resultados en tiempo real.


