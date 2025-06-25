import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo
modelo = joblib.load("C:/Users/mateo/OneDrive/Desktop/Trabajo Final FCE/rf_best.joblib")

# Diccionarios para traducir inputs legibles a números
sexo_dict = {"Masculino": 1, "Femenino": 2}
educacion_dict = {"Postgrado": 1, "Universitario": 2, "Secundario": 3, "Otro": 4}
estado_civil_dict = {"Casado/a": 1, "Soltero/a": 2, "Otro": 3}

# Título
st.title("Predicción de Default de Tarjeta de Crédito")

st.write("Completá los datos del cliente para predecir si hará default el próximo mes.")

# Inputs personales
limite_credito = st.number_input("Límite de crédito", min_value=0, step=10000)
sexo = st.selectbox("Sexo", list(sexo_dict.keys()))
educacion = st.selectbox("Nivel educativo", list(educacion_dict.keys()))
estado_civil = st.selectbox("Estado civil", list(estado_civil_dict.keys()))
edad = st.slider("Edad", 18, 100)

# Inputs de comportamiento de pago
st.markdown("### Comportamiento de pago (PAY_X) X= Meses de atraso")
pay_1 = st.selectbox("PAY_1", [ 0, 1, 2, 3, 4, 5, 6, 7, 8])
pay_2 = st.selectbox("PAY_2", [ 0, 1, 2, 3, 4, 5, 6, 7, 8])
pay_3 = st.selectbox("PAY_3", [ 0, 1, 2, 3, 4, 5, 6, 7, 8])
pay_4 = st.selectbox("PAY_4", [ 0, 1, 2, 3, 4, 5, 6, 7, 8])
pay_5 = st.selectbox("PAY_5", [ 0, 1, 2, 3, 4, 5, 6, 7, 8])
pay_6 = st.selectbox("PAY_6", [ 0, 1, 2, 3, 4, 5, 6, 7, 8])

# Inputs de montos facturados
st.markdown("### Resumen de la tarjeta (BILL_AMT) entre -400.000 y 1.000.000")
bill_amt1 = st.number_input("BILL_AMT1", step=100.0)
bill_amt2 = st.number_input("BILL_AMT2", step=100.0)
bill_amt3 = st.number_input("BILL_AMT3", step=100.0)
bill_amt4 = st.number_input("BILL_AMT4", step=100.0)
bill_amt5 = st.number_input("BILL_AMT5", step=100.0)
bill_amt6 = st.number_input("BILL_AMT6", step=100.0)

# Inputs de pagos realizados
st.markdown("### Montos pagados (PAY_AMT) entre 0 y 900.000")
pay_amt1 = st.number_input("PAY_AMT1", step=100.0)
pay_amt2 = st.number_input("PAY_AMT2", step=100.0)
pay_amt3 = st.number_input("PAY_AMT3", step=100.0)
pay_amt4 = st.number_input("PAY_AMT4", step=100.0)
pay_amt5 = st.number_input("PAY_AMT5", step=100.0)
pay_amt6 = st.number_input("PAY_AMT6", step=100.0)

# Botón para predecir
if st.button("Predecir"):
    input_df = pd.DataFrame([{
        'LIMIT_BAL': limite_credito,
        'SEX': sexo_dict[sexo],
        'EDUCATION': educacion_dict[educacion],
        'MARRIAGE': estado_civil_dict[estado_civil],
        'AGE': edad,
        'PAY_1': pay_1,
        'PAY_2': pay_2,
        'PAY_3': pay_3,
        'PAY_4': pay_4,
        'PAY_5': pay_5,
        'PAY_6': pay_6,
        'BILL_AMT1': bill_amt1,
        'BILL_AMT2': bill_amt2,
        'BILL_AMT3': bill_amt3,
        'BILL_AMT4': bill_amt4,
        'BILL_AMT5': bill_amt5,
        'BILL_AMT6': bill_amt6,
        'PAY_AMT1': pay_amt1,
        'PAY_AMT2': pay_amt2,
        'PAY_AMT3': pay_amt3,
        'PAY_AMT4': pay_amt4,
        'PAY_AMT5': pay_amt5,
        'PAY_AMT6': pay_amt6,
    }])

proba = modelo.predict_proba(input_df)[0]  # devuelve [P(clase 0), P(clase 1)]
pred = modelo.predict(input_df)[0]

# Mostrar resultado con probabilidad
if pred == 1:
    st.error(f"⚠️ Este cliente probablemente hará default el próximo mes.\n\nProbabilidad: {proba[1]*100:.2f}%")
else:
    st.success(f"✅ Este cliente probablemente no hará default.\n\nProbabilidad: {proba[0]*100:.2f}%")


