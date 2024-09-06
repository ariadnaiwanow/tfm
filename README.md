
PREDICCIÓN DE PROPENSIÓN A LA COMPRA DE PLANES DE PENSIONES
===========================================================

Descripción del Proyecto
-----------------------------------
Este proyecto tiene como objetivo predecir la propensión de los clientes a adquirir el servicio "Planes de Pensiones". El análisis se basa en datos históricos de ventas de easyMoney, una empresa financiera, con el fin de enfocar las próximas acciones de marketing y optimizar los recursos disponibles.

Contexto del Negocio
-----------------------------------
easyMoney se enfrenta al desafío de mejorar la rentabilidad de su base de clientes actual, con el fin de aumentar la penetración de productos financieros, especialmente los planes de pensiones, que se destacan por su rentabilidad. Este proyecto busca identificar a los clientes más propensos a adquirir este producto para focalizar las campañas de marketing y maximizar el impacto sin aumentar significativamente los costos de adquisición de nuevos clientes.

Análisis Descriptivo
-----------------------------------
Primero, realicé un análisis descriptivo de los datos utilizando Power BI. Se creó un dashboard que permite visualizar los indicadores clave de rendimiento (KPIs) y realizar un análisis evolutivo de las ventas y la rentabilidad de los productos de easyMoney. Este análisis mostró que los "Planes de Pensiones" eran el producto más rentable con el menor esfuerzo comercial, por lo que se decidió enfocar las próximas campañas en este servicio.
Dashboard: https://app.powerbi.com/view?r=eyJrIjoiMTgzMWJjMGMtODVkMi00NmQwLWIyMDYtZjU1ZWZkMDI0Yjc2IiwidCI6ImM0MGJlNDg3LWE3MTYtNDU5Ni05YTY1LTcyOWNlMjZmYzM1NiIsImMiOjR9

Análisis Predictivo
-----------------------------------
Para el análisis predictivo, utilicé un enfoque de Machine Learning que me permitió identificar a los clientes con mayor probabilidad de adquirir un "Plan de Pensiones". El desarrollo del modelo se llevó a cabo en Google Colab, conectando Google Drive para gestionar los datos almacenados en formato CSV.
Se probaron varios modelos de clasificación para predecir la probabilidad de que un cliente adquiera un plan de pensiones, eligiendo finalmente el Gradient Boosting Classifier por su rendimiento equilibrado entre precisión en entrenamiento y en prueba.

Datos Utilizados
-----------------------------------
Los datos utilizados en este proyecto provienen de la base de datos de easyMoney. Estos incluyen información histórica sobre los clientes, productos adquiridos y comportamiento de compra. Fueron procesados y limpiados para garantizar su calidad y facilitar su uso en los modelos predictivos.

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
    - accuracy_score: Para medir la precisión del modelo.
    - roc_auc_score: Para evaluar el área bajo la curva ROC.
    - roc_curve: Para generar las curvas ROC.
    - confusion_matrix: Para analizar los resultados de clasificación.

  * Evaluación Cruzada:
    - cross_val_score: Para realizar validación cruzada y evaluar el rendimiento del modelo en diferentes particiones de los datos.
   
Selección del Modelo
-----------------------------------
Se evaluaron seis modelos de clasificación y se decidió utilizar Gradient Boosting Classifier debido a su equilibrio entre la precisión en los datos de entrenamiento y los de prueba. Se realizaron múltiples iteraciones en el proceso de preprocesamiento de los datos para optimizar el rendimiento del modelo.

Aplicación Web
-----------------------------------
Finalmente, desarrollé una aplicación web interactiva utilizando Streamlit, que permite a los usuarios probar el modelo con distintos perfiles de clientes y visualizar los resultados de predicción en tiempo real. Esta herramienta facilita la toma de decisiones para el equipo de marketing de easyMoney, permitiendo ajustar las campañas basadas en los clientes más propensos a adquirir el producto.
https://ari-tfm-easy-money.streamlit.app/

Impacto Esperado
-----------------------------------
El modelo predictivo ayudará a easyMoney a enfocar mejor sus esfuerzos de marketing, identificando a los clientes más propensos a adquirir un plan de pensiones. Esto permitirá optimizar los recursos de marketing, maximizando el retorno sobre la inversión y mejorando la rentabilidad de la compañía.
