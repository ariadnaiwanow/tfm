import streamlit as st
import pickle 
import numpy as np 

def load_scaler():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return scaler

def load_model():
    gb_model = pickle.load(open("gradient_boosting_model.pkl", "rb"))
    return gb_model

def potencial_prediction(data):
    scaler = load_scaler()
    
    # Verificar que los datos tengan el tamaño correcto
    if len(data) != 35:
        return {"error": "La entrada debe contener exactamente 35 valores."}

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

# Tarjeta de débito
debit_card = st.checkbox("Tarjeta de Débito", value=False)
em_acount = st.checkbox("Cuenta EMC", value=False)

# Género
gender = st.selectbox("Género:", ["Seleccionar", "Hombre", "Mujer"])
gender_H = int(gender == "Hombre")
gender_V = int(gender == "Mujer")

# Canales de Entrada
entry_channels = st.selectbox(
    "Canales de Entrada:",
    ["KAT", "KFC", "KHK", "KHM", "KHN", "KHQ", "Otros"]
)
entry_channel_KAT = int(entry_channels == "KAT")
entry_channel_KFC = int(entry_channels == "KFC")
entry_channel_KHK = int(entry_channels == "KHK")
entry_channel_KHM = int(entry_channels == "KHM")
entry_channel_KHN = int(entry_channels == "KHN")
entry_channel_KHQ = int(entry_channels == "KHQ")
entry_channel_others = int(entry_channels == "Otros")

# Segmentos
segments = st.selectbox(
    "Segmentos:",
    ["01 - TOP", "02 - PARTICULARES", "03 - UNIVERSITARIO"]
)
segment_01_TOP = int(segments == "01 - TOP")
segment_02_PARTICULARES = int(segments == "02 - PARTICULARES")
segment_03_UNIVERSITARIO = int(segments == "03 - UNIVERSITARIO")

# Códigos de Región
region_codes = st.selectbox(
    "Códigos de Región:",
    ["3.0", "8.0", "11.0", "14.0", "15.0", "18.0", "28.0", "29.0", "30.0", "33.0", 
     "35.0", "36.0", "39.0", "41.0", "45.0", "46.0", "47.0", "50.0", "Otros"]
)
region_code_3_0 = int(region_codes == "3.0")
region_code_8_0 = int(region_codes == "8.0")
region_code_11_0 = int(region_codes == "11.0")
region_code_14_0 = int(region_codes == "14.0")
region_code_15_0 = int(region_codes == "15.0")
region_code_18_0 = int(region_codes == "18.0")
region_code_28_0 = int(region_codes == "28.0")
region_code_29_0 = int(region_codes == "29.0")
region_code_30_0 = int(region_codes == "30.0")
region_code_33_0 = int(region_codes == "33.0")
region_code_35_0 = int(region_codes == "35.0")
region_code_36_0 = int(region_codes == "36.0")
region_code_39_0 = int(region_codes == "39.0")
region_code_41_0 = int(region_codes == "41.0")
region_code_45_0 = int(region_codes == "45.0")
region_code_46_0 = int(region_codes == "46.0")
region_code_47_0 = int(region_codes == "47.0")
region_code_50_0 = int(region_codes == "50.0")
region_code_others = int(region_codes == "Otros")

# Valores numéricos
age = st.number_input("Edad", min_value=0, value=0)
salary = st.number_input("Salario", min_value=0, value=0)

# Convertir edad y salario a log
log_age = np.log10(age + 1)  # Se suma 1 para evitar log(0)
log_salary = np.log10(salary + 1)  # Se suma 1 para evitar log(0)

# Cliente Activo
active_customer = st.selectbox("Cliente Activo", ["No", "Sí"])
active_customer_val = 1 if active_customer == "Sí" else 0

# Crear la lista de características para el modelo
features = [
    gender_H, active_customer_val, int(debit_card), int(em_acount), entry_channel_KAT, 
    entry_channel_KFC, entry_channel_KHK, entry_channel_KHM, entry_channel_KHN, 
    entry_channel_KHQ, entry_channel_others, segment_01_TOP, segment_02_PARTICULARES, 
    segment_03_UNIVERSITARIO, region_code_3_0, region_code_8_0, region_code_11_0, 
    region_code_14_0, region_code_15_0, region_code_18_0, region_code_28_0, 
    region_code_29_0, region_code_30_0, region_code_33_0, region_code_35_0, 
    region_code_36_0, region_code_39_0, region_code_41_0, region_code_45_0, 
    region_code_46_0, region_code_47_0, region_code_50_0, region_code_others, 
    log_age, log_salary
]

# Botón para realizar la predicción
if st.button("Analizar"):
    result = potencial_prediction(features)
    if 'error' in result:
        st.error(result['error'])
    else:
        st.subheader("Resultado:")
        st.info("El resultado es: " + result + ".")
