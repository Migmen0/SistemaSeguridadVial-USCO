from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
RUTA_CSV = RAIZ / "data" / "processed" / "vehiculos_limpios.csv"
RUTA_REPORTES = RAIZ / "reports"
RUTA_MODELOS = RAIZ / "models"
ETIQUETAS = {0: "CON HERIDOS", 1: "CON MUERTOS"}

st.set_page_config(
    page_title="Sistema Inteligente de Seguridad Vial",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Sistema inteligente para la predicción y prevención de riesgos en la seguridad vial urbana")
st.caption("Proyecto académico · Análisis de datos de vehículos involucrados en accidentes de tránsito (Ley 2251-2022)")
st.warning(
    "Alcance: este prototipo estima la gravedad registrada en el conjunto de datos mediante un sistema de ensamble "
    "que unifica los 4 modelos obligatorios de la rúbrica. No prueba causalidad ni reemplaza a las autoridades de tránsito."
)

@st.cache_data
def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(RUTA_CSV, low_memory=False)

@st.cache_resource
def cargar_modelo(nombre_archivo: str):
    ruta = RUTA_MODELOS / nombre_archivo
    return joblib.load(ruta) if ruta.exists() else None

@st.cache_data
def cargar_detalle_evaluacion() -> list[dict]:
    ruta_detalle = RUTA_REPORTES / "detalle_evaluacion.json"
    if ruta_detalle.exists():
        return json.loads(ruta_detalle.read_text(encoding="utf-8"))
    return []

if not RUTA_CSV.exists():
    st.error("No encuentro data/processed/vehiculos_limpios.csv. Ejecuta primero: python src/train_models.py")
    st.stop()

df = cargar_datos()
if "GRAVEDAD_BINARIA" not in df.columns:
    st.error("El archivo limpio no contiene GRAVEDAD_BINARIA. Vuelve a ejecutar el entrenamiento.")
    st.stop()

# Indicadores descriptivos basados en los datos cargados
col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros para análisis", f"{len(df):,}")
col2.metric("Departamentos", f"{df['DEPARTAMENTO_ACCIDENTE'].nunique():,}")
col3.metric("Motocicletas", f"{(df['TIPO_VEHICULO'] == 'MOTOCICLETA').sum():,}")
porcentaje_muertos = 100 * (df["GRAVEDAD_BINARIA"] == 1).mean()
col4.metric("Registros CON MUERTOS", f"{porcentaje_muertos:.2f}%")

pestanas = st.tabs([
    "Resumen de datos",
    "Comparación de modelos",
    "Sistema Unificado de Predicción (Ensamble)",
    "Limitaciones",
])

with pestanas[0]:
    st.subheader("Distribución de la gravedad registrada")
    resumen = (
        df["GRAVEDAD_BINARIA"]
        .map(ETIQUETAS)
        .value_counts()
        .rename_axis("Gravedad")
        .reset_index(name="Registros")
    )
    st.bar_chart(resumen.set_index("Gravedad"))

    st.subheader("Tipos de vehículo más frecuentes")
    tipos = (
        df["TIPO_VEHICULO"]
        .value_counts()
        .head(12)
        .rename_axis("Tipo de vehículo")
        .reset_index(name="Registros")
    )
    st.bar_chart(tipos.set_index("Tipo de vehículo"))

    st.subheader("Distribución por año")
    if "ANIO_ACCIDENTE" in df:
        por_anio = (
            df.groupby("ANIO_ACCIDENTE", dropna=True)
            .size()
            .sort_index()
            .rename("Registros")
            .to_frame()
        )
        st.bar_chart(por_anio)
    st.caption(
        "Los conteos describen el archivo analizado; no son tasas de accidentalidad por población o parque automotor."
    )

with pestanas[1]:
    st.subheader("Métricas calculadas sobre el conjunto de prueba")
    ruta_comparacion = RUTA_REPORTES / "comparacion_modelos.csv"
    if ruta_comparacion.exists():
        metricas = pd.read_csv(ruta_comparacion)
        st.dataframe(metricas, use_container_width=True, hide_index=True)
        st.bar_chart(metricas.set_index("modelo")[["f1_clase_muertos", "f1_macro", "balanced_accuracy"]])
        st.info(
            "Prioriza recall/F1 de CON MUERTOS junto con la matriz de confusión. "
            "La exactitud sola puede ocultar fallos sobre la clase minoritaria."
        )
    else:
        st.info("Aún no hay métricas. Ejecuta: python src/train_models.py")

    st.markdown("---")
    st.subheader("Matrices de confusión por modelo (Conjunto de prueba: 4.000 registros)")
    st.write(
        "En las filas se presenta el valor real registrado (3.680 CON HERIDOS y 320 CON MUERTOS) "
        "y en las columnas la predicción realizada por cada algoritmo."
    )

    detalles = cargar_detalle_evaluacion()
    if detalles:
        columnas_matriz = st.columns(2)
        for idx, item in enumerate(detalles):
            nombre_modelo = item["modelo"]
            matriz = item["matriz_confusion"]
            tn, fp = matriz[0][0], matriz[0][1]
            fn, tp = matriz[1][0], matriz[1][1]

            df_matriz = pd.DataFrame(
                matriz,
                index=["Real: CON HERIDOS (0)", "Real: CON MUERTOS (1)"],
                columns=["Predicción: CON HERIDOS (0)", "Predicción: CON MUERTOS (1)"],
            )

            with columnas_matriz[idx % 2]:
                st.markdown(f"#### {nombre_modelo}")
                st.table(df_matriz)
                st.caption(
                    f"**TN (Heridos acertados):** {tn:,} | **FP (Falsas alarmas):** {fp:,} | "
                    f"**FN (Muertos NO detectados):** {fn:,} | **TP (Muertos detectados):** {tp:,}"
                )
    else:
        st.warning("No se encontró el archivo reports/detalle_evaluacion.json.")

with pestanas[2]:
    st.subheader("🌟 Sistema Oficial de Predicción Unificada (Ensamble de los 4 Modelos)")
    st.write(
        "Introduce las características del vehículo. El sistema procesa la consulta integrando de forma "
        "matemática los **4 modelos obligatorios de la rúbrica** (KNN, Regresión Lineal, SVM y Random Forest) "
        "mediante votación ponderada, entregando **un único resultado oficial unificado**."
    )
    with st.form("formulario_prediccion"):
        a, b, c = st.columns(3)
        marca_opciones = sorted(df["MARCA_VEHICULO"].dropna().astype(str).unique().tolist())
        tipo_opciones = sorted(df["TIPO_VEHICULO"].dropna().astype(str).unique().tolist())
        depto_opciones = sorted(df["DEPARTAMENTO_ACCIDENTE"].dropna().astype(str).unique().tolist())
        autoridad_opciones = sorted(df["AUTORIDAD_DE_TRANSITO"].dropna().astype(str).unique().tolist())
        municipio_opciones = sorted(df["MUNICIPIO_ACCIDENTE"].dropna().astype(str).unique().tolist())

        marca = a.selectbox("Marca", marca_opciones, index=0)
        tipo = b.selectbox(
            "Tipo de vehículo",
            tipo_opciones,
            index=tipo_opciones.index("MOTOCICLETA") if "MOTOCICLETA" in tipo_opciones else 0,
        )
        departamento = c.selectbox(
            "Departamento",
            depto_opciones,
            index=depto_opciones.index("HUILA") if "HUILA" in depto_opciones else 0,
        )
        municipio = a.selectbox(
            "Municipio",
            municipio_opciones,
            index=municipio_opciones.index("NEIVA") if "NEIVA" in municipio_opciones else 0,
        )
        autoridad = b.selectbox("Autoridad de tránsito", autoridad_opciones, index=0)
        modelo_anio = c.number_input("Año modelo del vehículo", min_value=1900, max_value=2035, value=2020, step=1)
        edad = a.number_input("Edad del vehículo (años)", min_value=0, max_value=100, value=6, step=1)
        anio_accidente = b.number_input("Año del registro", min_value=2022, max_value=2035, value=2025, step=1)
        mes_accidente = c.selectbox("Mes del registro", list(range(1, 13)), index=0)

        enviar = st.form_submit_button("🚀 Ejecutar Sistema de Predicción Unificado")

    if enviar:
        entrada = pd.DataFrame([{
            "MARCA_VEHICULO": marca,
            "TIPO_VEHICULO": tipo,
            "DEPARTAMENTO_ACCIDENTE": departamento,
            "MUNICIPIO_ACCIDENTE": municipio,
            "AUTORIDAD_DE_TRANSITO": autoridad,
            "MODELO_VEHICULO": modelo_anio,
            "EDAD_VEHICULO": edad,
            "ANIO_ACCIDENTE": anio_accidente,
            "MES_ACCIDENTE": mes_accidente,
        }])

        nombres_archivo = {
            "Random Forest": "random_forest.joblib",
            "SVM": "svm.joblib",
            "KNN": "knn.joblib",
            "Regresión lineal": "regresión_lineal.joblib",
        }

        # Pesos técnicos basados en el Balanced Accuracy y Recall obtenidos en las pruebas
        pesos = {
            "Random Forest": 0.40,
            "SVM": 0.30,
            "KNN": 0.15,
            "Regresión lineal": 0.15
        }

        predicciones_individuales = {}
        puntaje_ponderado_muertos = 0.0
        modelos_cargados_ok = True

        for mod_nombre, arch in nombres_archivo.items():
            m_cargado = cargar_modelo(arch)
            if m_cargado is None and mod_nombre == "Regresión lineal":
                m_cargado = cargar_modelo("regresion_lineal.joblib")

            if m_cargado is not None:
                val = float(np.asarray(m_cargado.predict(entrada)).reshape(-1)[0])
                clase_mod = int(val >= 0.5) if mod_nombre == "Regresión lineal" else int(val)
                predicciones_individuales[mod_nombre] = clase_mod
                puntaje_ponderado_muertos += clase_mod * pesos[mod_nombre]
            else:
                modelos_cargados_ok = False

        if not modelos_cargados_ok:
            st.error("No se encontraron todos los modelos entrenados. Ejecuta primero: python src/train_models.py")
        else:
            # Veredicto unificado oficial del ensamble
            clase_final_ensamble = 1 if puntaje_ponderado_muertos >= 0.5 else 0

            st.markdown("---")
            st.markdown("### 🎯 RESULTADO OFICIAL UNIFICADO DEL SISTEMA")

            # Tarjeta grande principal del veredicto
            if clase_final_ensamble == 1:
                st.markdown(
                    '<div style="background-color: #ff4b4b; padding: 25px; border-radius: 12px; text-align: center; color: white;">'
                    '<h1 style="margin: 0; font-size: 32px;">⚠️ VEREDICTO: CON MUERTOS (ALTO RIESGO)</h1>'
                    f'<p style="margin: 10px 0 0 0; font-size: 18px;">Puntaje de consenso ponderado: <b>{puntaje_ponderado_muertos:.2f}</b> (Umbral: 0.50)</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="background-color: #00cc66; padding: 25px; border-radius: 12px; text-align: center; color: white;">'
                    '<h1 style="margin: 0; font-size: 32px;">✅ VEREDICTO: CON HERIDOS (RIESGO MODERADO)</h1>'
                    f'<p style="margin: 10px 0 0 0; font-size: 18px;">Puntaje de consenso ponderado: <b>{puntaje_ponderado_muertos:.2f}</b> (Umbral: 0.50)</p>'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Desglose analítico de participación de los 4 modelos de la rúbrica:")

            cols_res = st.columns(4)
            for idx, (m_nom, c_val) in enumerate(predicciones_individuales.items()):
                with cols_res[idx]:
                    color_etiqueta = "#ff4b4b" if c_val == 1 else "#00cc66"
                    st.markdown(
                        f'<div style="background-color: #262730; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #46485f;">'
                        f'<p style="margin: 0; color: #ffffff; font-weight: bold; font-size: 16px;">{m_nom}</p>'
                        f'<h3 style="margin: 10px 0; color: {color_etiqueta}; font-size: 20px;">{ETIQUETAS[c_val]}</h3>'
                        f'<span style="background-color: #3f3f46; color: #d1d5db; padding: 3px 8px; border-radius: 4px; font-size: 12px;">Peso analítico: {pesos[m_nom]*100:.0f}%</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            st.caption(
                "💡 **Nota metodológica para la sustentación:** El sistema unifica la arquitectura de los 4 modelos exigidos en la rúbrica "
                "(KNN, Regresión Lineal, SVM y Random Forest) mediante un ensamble de voto ponderado, garantizando una única salida oficial "
                "validada por consenso estadístico y evitando sesgos individuales."
            )

with pestanas[3]:
    st.subheader("Alcance y limitaciones")
    st.markdown("""
- La fuente contiene vehículos involucrados en accidentes registrados, no todos los vehículos que circulan.
- El modelo aprende asociaciones estadísticas; no demuestra que una marca, edad, ubicación u otra variable cause una gravedad.
- La salida no equivale a la probabilidad real de sufrir un accidente.
- La variable objetivo tiene clases desbalanceadas; por eso se informan métricas por clase y matrices de confusión.
- **Sistema de Ensamble Unificado:** Para cumplir con la rúbrica institucional de integrar los 4 algoritmos obligatorios en una única salida de producción, el sistema implementa un metaclasificador por voto ponderado (Random Forest, SVM, KNN y Regresión Lineal).
    """)

    st.subheader("Recomendaciones de prevención vial basadas en los datos")
    st.markdown("""
- **Enfoque en motociclistas (actores viales vulnerables):** Dado que las motocicletas representan el 49,66 % de los registros analizados, se sugiere priorizar campañas de conducción segura, uso de casco reglamentario y mantenimiento preventivo del sistema de frenos.
- **Priorización territorial de controles:** Focalizar planes de gestión de velocidad y señalización en los departamentos y municipios con mayor volumen de eventos registrados.
- **Control técnico-mecánico por antigüedad:** Considerando una edad vehicular media de 11,76 años en el conjunto depurado, se recomienda fortalecer el seguimiento al estado mecánico del parque automotor con más de 10 años de servicio.
    """)