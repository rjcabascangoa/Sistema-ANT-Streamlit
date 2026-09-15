import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la interfaz
st.set_page_config(page_title="Sistema Anti-Siniestralidad ANT", layout="wide")

st.title("🚨 Sistema Inteligente de Predicción de Siniestros Viales (ANT / ECU 911)")
st.markdown("Herramienta interactiva para la estimación de riesgo y severidad de accidentes en las vías de Ecuador.")

# Cargar el modelo y procesadores guardados
@st.cache_resource
def cargar_recursos():
    model = joblib.load('modelo_siniestros.pkl')
    cols = joblib.load('columnas_siniestros.pkl')
    encoders = joblib.load('encoders_siniestros.pkl')
    scaler = joblib.load('scaler_siniestros.pkl')
    return model, cols, encoders, scaler

try:
    model, cols, encoders, scaler = cargar_recursos()

    st.sidebar.header("📋 Parámetros del Siniestro")
    
    # Formulario interactivo
    provincia = st.sidebar.selectbox("Provincia", encoders['provincia'].classes_)
    jornada = st.sidebar.selectbox("Jornada", encoders['jornada'].classes_)
    tipo_vehiculo = st.sidebar.selectbox("Tipo de Vehículo", encoders['tipo_vehiculo'].classes_)
    causa_probable = st.sidebar.selectbox("Causa Probable", encoders['causa_probable'].classes_)
    clima = st.sidebar.selectbox("Clima", encoders['clima'].classes_)
    zona = st.sidebar.selectbox("Zona", encoders['zona'].classes_)
    edad_conductor = st.sidebar.slider("Edad del Conductor", 18, 75, 30)

    # Botón de simulación
    if st.sidebar.button("Calcular Nivel de Riesgo"):
        input_dict = {
            'provincia': encoders['provincia'].transform([provincia])[0],
            'jornada': encoders['jornada'].transform([jornada])[0],
            'tipo_vehiculo': encoders['tipo_vehiculo'].transform([tipo_vehiculo])[0],
            'causa_probable': encoders['causa_probable'].transform([causa_probable])[0],
            'clima': encoders['clima'].transform([clima])[0],
            'zona': encoders['zona'].transform([zona])[0],
            'edad_conductor': edad_conductor
        }
        
        input_df = pd.DataFrame([input_dict])
        input_df['edad_conductor'] = scaler.transform(input_df[['edad_conductor']])
        input_df = input_df[cols]
        
        prediccion = model.predict(input_df)[0]
        probabilidad = model.predict_proba(input_df)[0][1]

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Clasificación de Severidad")
            if prediccion == 1:
                st.error("⚠️ **SEVERIDAD ALTA** (Riesgo elevado de heridos o fallecidos)")
            else:
                st.success("✅ **SEVERIDAD BAJA** (Predominio de solo daños materiales)")
        
        with col2:
            st.subheader("Nivel de Probabilidad")
            st.metric(label="Probabilidad de Severidad Alta", value=f"{probabilidad * 100:.1f}%")

        st.markdown("---")
        st.subheader("💡 Recomendación Operativa para la ANT / ECU 911")
        if probabilidad >= 0.60:
            st.warning("Nivel crítico: Se recomienda coordinar un operativo preventivo inmediato, control de velocidad con fotorradar y patrullaje prioritario.")
        else:
            st.info("Nivel controlado: Mantener monitoreo de rutina e inspección habitual del tránsito.")

except Exception as e:
    st.error(f"Error al cargar los archivos: {e}")
    st.info("Verifica que los archivos .pkl estén en la misma carpeta.")