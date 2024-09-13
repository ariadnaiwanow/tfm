import streamlit as st
import pickle 
import numpy as np 
from datetime import datetime

def load_scaler():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return scaler

def load_model():
    gb_model = pickle.load(open("gradient_boosting_model.pkl", "rb"))
    return gb_model

def potencial_prediction(data):
    scaler = load_scaler()

    # Verificar que los datos tengan el tamaño correcto													  
    if len(data) != 45:  # Cambiamos a 45, porque el modelo espera 46 características
        return {"error": f"La entrada debe contener exactamente 45 valores, pero tiene {len(data)} valores."}

    # Convertir la lista de entrada a un array numpy y escalarla
    data_array = np.array([data])  # Convertir a array 2D para el scaler
    
    try:
        data_scaled = scaler.transform(data_array)
    except Exception as e:
        return {"error": f"Error al transformar los datos: {str(e)}"}															 

    # Cargar el modelo
    gb_model = load_model()
    
    # Predecir la clase
    try:
        prediction = gb_model.predict(data_scaled)
    except Exception as e:
        return {"error": f"Error al predecir la clase: {str(e)}"}																 

    # Mapear la clase predicha a una cadena descriptiva
    class_name = "ALTA probabilidad de adquirir servicio -Pension Plan-" if prediction == 1 else "BAJA probabilidad de adquirir servicio -Pension Plan-"

    return class_name

st.header("Potenciales clientes -Pension Plan- App")

st.subheader("Ingresar Características del Cliente")

# Crear la interfaz de entrada

# Tarjeta de débito y cuenta EMC
debit_card = st.checkbox("Tarjeta de Débito", value=False)
em_acount = st.checkbox("Cuenta EMC", value=False)

# Género
gender = st.selectbox("Género:", ["Seleccionar", "Hombre", "Mujer"])
gender_val = True if gender == "Hombre" else False

# Canales de Entrada
entry_channels = st.selectbox(
    "Canales de Entrada:",
    ["KAT", "KFC", "KHE", "KHK", "KHM", "KHN", "Otros"]
)
entry_channel_KAT = int(entry_channels == "KAT")
entry_channel_KFC = int(entry_channels == "KFC")
entry_channel_KHE = int(entry_channels == "KHE")
entry_channel_KHK = int(entry_channels == "KHK")
entry_channel_KHM = int(entry_channels == "KHM")
entry_channel_KHN = int(entry_channels == "KHN")
entry_channel_others = int(entry_channels == "Otros")

# Segmentos
segments = st.selectbox(
    "Segmentos:",
    ["01 - TOP", "02 - PARTICULARES", "03 - UNIVERSITARIO"]
)
segment_01_TOP = int(segments == "01 - TOP")
segment_02_PARTICULARES = int(segments == "02 - PARTICULARES")
segment_03_UNIVERSITARIO = int(segments == "03 - UNIVERSITARIO")

# Códigos de Región (usamos un selectbox en lugar de checkboxes)
region_codes = st.selectbox(
    "Código de Región:",
    ["2.0", "3.0", "6.0", "7.0", "8.0", "9.0", "11.0", "12.0", "13.0", "14.0", 
     "15.0", "17.0", "18.0", "21.0", "28.0", "29.0", "30.0", "33.0", "35.0", 
     "36.0", "37.0", "39.0", "41.0", "43.0", "45.0", "46.0", "47.0", "50.0", "Otros"]
)

# Mapear los códigos de región a valores binarios
region_code_map = {
    "2.0": 0, "3.0": 0, "6.0": 0, "7.0": 0, "8.0": 0, "9.0": 0, "11.0": 0, "12.0": 0, 
    "13.0": 0, "14.0": 0, "15.0": 0, "17.0": 0, "18.0": 0, "21.0": 0, "28.0": 0, 
    "29.0": 0, "30.0": 0, "33.0": 0, "35.0": 0, "36.0": 0, "37.0": 0, "39.0": 0, 
    "41.0": 0, "43.0": 0, "45.0": 0, "46.0": 0, "47.0": 0, "50.0": 0, "Otros": 0
}
region_code_map[region_codes] = 1  # Solo la región seleccionada será 1


# Edad y antigüedad
age = st.number_input("Edad", min_value=0, value=0)
log_log_age = np.log10(np.log10(age + 1) + 1)  # Usamos log(log(edad + 1))

# Fecha de alta en easyMoney (mes y año)
mes_alta = st.selectbox("Mes de Alta:", list(range(1, 13)))  # De 1 a 12
año_alta = st.number_input("Año de Alta:", min_value=1900, max_value=datetime.now().year, value=2000)

# Calcular días de antigüedad desde la fecha de alta hasta el día actual
fecha_alta = datetime(año_alta, mes_alta, 1)  # Se asume el primer día del mes
dias_antiguedad = (datetime.now() - fecha_alta).days

# Convertir los días de antigüedad a log-log
log_log_dias_antiguedad = np.log10(np.log10(dias_antiguedad + 1) + 1)

# Cliente Activo
active_customer = st.selectbox("Cliente Activo", ["Seleccionar", "Sí", "No"])
active_customer_val = 1 if active_customer == "Sí" else 0

# Crear la lista de características para el modelo
features = [
    gender_val, active_customer_val, int(debit_card), int(em_acount), 
    entry_channel_KAT, entry_channel_KFC, entry_channel_KHE, entry_channel_KHK, 
    entry_channel_KHM, entry_channel_KHN, entry_channel_others, 
    segment_01_TOP, segment_02_PARTICULARES, segment_03_UNIVERSITARIO, 
    *region_codes_map.values(),  # Agregamos todas las regiones
    log_log_age, log_log_dias_antiguedad
]

# Botón para realizar la predicción
if st.button("Analizar"):
    result = potencial_prediction(features)
    st.subheader("Resultado:")
    st.info("El resultado es: " + str(result) + ".")
