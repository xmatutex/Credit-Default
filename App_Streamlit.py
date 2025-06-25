import streamlit as st
import pandas as pd
import joblib

modelo = joblib.load("modelo_gradientboost.joblib") 

st.title("Predicción de Default de Tarjeta de Crédito")

st.markdown("### Ingresá los datos del cliente:")

# Inputs personales
limite_credito = st.number_input("Límite de crédito (Max $1.000.000)", min_value=0, step=10000)
sexo = st.selectbox("Sexo", options=[1, 2], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
educacion = st.selectbox("Nivel educativo", options=[1, 2, 3, 4],
                         format_func=lambda x: {1: "Postgrado", 2: "Universidad", 3: "Secundario", 4: "Otros"}[x])
estado_civil = st.selectbox("Estado civil", options=[1, 2, 3],
                            format_func=lambda x: {1: "Casado/a", 2: "Soltero/a", 3: "Otro"}[x])
edad = st.slider("Edad", 18, 100)

# Comportamiento de pago
st.markdown("### Estados de pago (0 = pagó en término, 1+ = atraso)")
pagos = {}
for i in range(1, 7):
    pagos[f"PAY_{i}"] = st.selectbox(f"Pago atrasado {i} m", list(range(0, 10)))

# Montos facturados
st.markdown("### Resumen de credito pro mes (promedio $40.000)")
bill_amts = {}
for i in range(1, 7):
    bill_amts[f"BILL_AMT{i}"] = st.number_input(f"Resumen mes {i}", step=100.0)

# Montos pagados
st.markdown("### Pagos realizados en meses anteriores (promedio $50.000)")
pay_amts = {}
for i in range(1, 6+1):
    pay_amts[f"PAY_AMT{i}"] = st.number_input(f"Pago mes {i}", step=100.0)

# Armar DataFrame
# Botón para predecir
if st.button("Predecir"):
    input_data = {
        "LIMIT_BAL": limite_credito,
        "SEX": sexo,
        "EDUCATION": educacion,
        "MARRIAGE": estado_civil,
        "AGE": edad
    }
    input_data.update(pagos)
    input_data.update(bill_amts)
    input_data.update(pay_amts)

    input_df = pd.DataFrame([input_data])

    # Predicción
    pred = modelo.predict(input_df)[0]
    proba = modelo.predict_proba(input_df)[0]

    # Mostrar resultado con probabilidad
    if pred == 1:
        st.error(f"⚠️ Este cliente probablemente **hará default**.\n\nProbabilidad: {proba[1]*100:.2f}%")
    else:
        st.success(f"✅ Este cliente probablemente **no hará default**.\n\nProbabilidad: {proba[0]*100:.2f}%")
