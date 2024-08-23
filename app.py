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
    if len(data) != 42:
        return {"error": "La entrada debe contener exactamente 42 valores."}

    # Convertir la lista de entrada a un array numpy y escalarla
    data_array = np.array([data])  # Convertir a array 2D para el scaler
    data_scaled = scaler.transform(data_array)

    # Cargar el modelo
    gb_model = load_model()
    
    # Predecir la clase
    prediction = gb_model.predict(data_scaled)

    # Mapear la clase predicha a una cadena descriptiva
    class_name = "ALTA probabilidad de adquirir servicio -Pension Plan-" if prediction == 1 else "BAJA probabilidad de adquirir servicio -Pension Plan-"

    return class_name

st.header("Potenciales clientes -Pension Plan- App")

st.subheader("Ingresar Características del Cliente")

# Crear la interfaz de entrada con menús desplegables y entradas numéricas

# Crear la interfaz de entrada
debit_card = st.checkbox("Debit Card", value=False)
em_acount = st.checkbox("EM Account", value=False)

# Género
gender = st.selectbox("Género:", ["Seleccionar", "Hombre", "Mujer"])
gender_H = int(gender == "Hombre")
gender_V = int(gender == "Mujer")

# Entry Channels
entry_channels = st.multiselect(
    "Canales de Entrada:",
    ["KAT", "KFA", "KFC", "KHE", "KHK", "KHN", "KHQ", "RED", "Otros"]
)
entry_channel_KAT = int("KAT" in entry_channels)
entry_channel_KFA = int("KFA" in entry_channels)
entry_channel_KFC = int("KFC" in entry_channels)
entry_channel_KHE = int("KHE" in entry_channels)
entry_channel_KHK = int("KHK" in entry_channels)
entry_channel_KHN = int("KHN" in entry_channels)
entry_channel_KHQ = int("KHQ" in entry_channels)
entry_channel_RED = int("RED" in entry_channels)
entry_channel_others = int("Otros" in entry_channels)

# Segmentos
segments = st.multiselect(
    "Segmentos:",
    ["01 - TOP", "02 - PARTICULARES", "03 - UNIVERSITARIO"]
)
segment_01_TOP = int("01 - TOP" in segments)
segment_02_PARTICULARES = int("02 - PARTICULARES" in segments)
segment_03_UNIVERSITARIO = int("03 - UNIVERSITARIO" in segments)

# Region Codes
region_codes = st.multiselect(
    "Códigos de Región:",
    ["2.0", "3.0", "6.0", "7.0", "8.0", "11.0", "12.0", "14.0", "15.0", "18.0", "28.0", "29.0", "30.0", "33.0", "35.0", "36.0", "37.0", "39.0", "41.0", "45.0", "46.0", "47.0", "50.0", "Otros"]
)
region_code_2_0 = int("2.0" in region_codes)
region_code_3_0 = int("3.0" in region_codes)
region_code_6_0 = int("6.0" in region_codes)
region_code_7_0 = int("7.0" in region_codes)
region_code_8_0 = int("8.0" in region_codes)
region_code_11_0 = int("11.0" in region_codes)
region_code_12_0 = int("12.0" in region_codes)
region_code_14_0 = int("14.0" in region_codes)
region_code_15_0 = int("15.0" in region_codes)
region_code_18_0 = int("18.0" in region_codes)
region_code_28_0 = int("28.0" in region_codes)
region_code_29_0 = int("29.0" in region_codes)
region_code_30_0 = int("30.0" in region_codes)
region_code_33_0 = int("33.0" in region_codes)
region_code_35_0 = int("35.0" in region_codes)
region_code_36_0 = int("36.0" in region_codes)
region_code_37_0 = int("37.0" in region_codes)
region_code_39_0 = int("39.0" in region_codes)
region_code_41_0 = int("41.0" in region_codes)
region_code_45_0 = int("45.0" in region_codes)
region_code_46_0 = int("46.0" in region_codes)
region_code_47_0 = int("47.0" in region_codes)
region_code_50_0 = int("50.0" in region_codes)
region_code_others = int("Otros" in region_codes)

# Valores numéricos
age = st.number_input("Edad", min_value=0, value=0)
salary = st.number_input("Salario", min_value=0, value=0)

# Convertir edad y salario a log
log_age = np.log10(age + 1)  # Se suma 1 para evitar log(0)
log_salary = np.log10(salary + 1)  # Se suma 1 para evitar log(0)

# Crear la lista de características para el modelo
features = [
    int(debit_card), int(em_acount), gender_H, gender_V, entry_channel_KAT, entry_channel_KFA,
    entry_channel_KFC, entry_channel_KHE, entry_channel_KHK, entry_channel_KHN,
    entry_channel_KHQ, entry_channel_RED, entry_channel_others, segment_01_TOP,
    segment_02_PARTICULARES, segment_03_UNIVERSITARIO, region_code_2_0, region_code_3_0,
    region_code_6_0, region_code_7_0, region_code_8_0, region_code_11_0, region_code_12_0,
    region_code_14_0, region_code_15_0, region_code_18_0, region_code_28_0, region_code_29_0,
    region_code_30_0, region_code_33_0, region_code_35_0, region_code_36_0, region_code_37_0,
    region_code_39_0, region_code_41_0, region_code_45_0, region_code_46_0, region_code_47_0,
    region_code_50_0, region_code_others, log_age, log_salary
]

# Botón para realizar la predicción
if st.button("Analizar"):
    result = potencial_prediction(features)
    st.subheader("Resultado:")
    st.info("El resultado es: " + result + ".")
