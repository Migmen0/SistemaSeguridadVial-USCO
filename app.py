"""
Módulo principal de la aplicación web interactiva en Streamlit.

Propósito:
    Proporcionar una interfaz gráfica para consultar estadísticas descriptivas,
    visualizar gráficos del análisis exploratorio, comparar el desempeño de los
    cuatro modelos de aprendizaje automático (incluyendo sus matrices de confusión),
    ejecutar demostraciones de predicción y documentar las limitaciones del sistema.

Responsabilidades:
    - Cargar en caché el conjunto de datos depurado y los pipelines entrenados.
    - Presentar indicadores generales y gráficos del análisis exploratorio.
    - Mostrar métricas comparativas y las matrices de confusión de los 4 modelos.
    - Realizar demostraciones de predicción usando el preprocesamiento entrenado.
    - Explicar las limitaciones metodológicas y recomendaciones de prevención vial.

Entradas:
    - data/processed/vehiculos_limpios.csv
    - reports/comparacion_modelos.csv
    - reports/detalle_evaluacion.json
    - models/*.joblib

Salidas:
    - Aplicación web interactiva desplegada localmente en el navegador.

Dependencias relevantes:
    - streamlit, pandas, numpy, joblib, pathlib, json.
"""

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
    "Alcance: este prototipo estima la gravedad registrada en el conjunto de datos. "
    "No calcula la probabilidad real de sufrir un accidente, no prueba causalidad y no reemplaza a las autoridades de tránsito."
)


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    """
    Carga el conjunto limpio generado por el flujo de preparación de datos.

    Descripción:
        Lee el archivo CSV procesado desde disco y lo mantiene en memoria caché.

    Parámetros:
        Ninguno.

    Valor de retorno:
        pd.DataFrame: DataFrame con los 290.936 registros depurados.

    Excepciones relevantes:
        FileNotFoundError: Si el archivo CSV procesado no existe en la ruta esperada.
    """
    return pd.read_csv(RUTA_CSV, low_memory=False)


@st.cache_resource
def cargar_modelo(nombre_archivo: str):
    """
    Carga un Pipeline serializado evitando recargarlo o reentrenarlo en cada interacción.

    Descripción:
        Recupera desde la carpeta models/ el objeto Pipeline que contiene tanto el
        preprocesador (ColumnTransformer) como el algoritmo entrenado.

    Parámetros:
        nombre_archivo (str): Nombre del archivo .joblib a cargar.

    Valor de retorno:
        Pipeline | None: Modelo listo para predecir o None si no se encuentra el archivo.
    """
    ruta = RUTA_MODELOS / nombre_archivo
    return joblib.load(ruta) if ruta.exists() else None


@st.cache_data
def cargar_detalle_evaluacion() -> list[dict]:
    """
    Carga el archivo JSON con las matrices de confusión y métricas por clase.

    Descripción:
        Lee el reporte detallado generado durante la evaluación sobre el conjunto
        de prueba (4.000 registros).

    Parámetros:
        Ninguno.

    Valor de retorno:
        list[dict]: Lista con la matriz de confusión y reporte de cada uno de los 4 modelos.
    """
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
    "Demostración de predicción",
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
    st.subheader("Formulario de demostración")
    st.write(
        "Introduce características de un vehículo y del registro. El resultado es una estimación experimental de la gravedad registrada."
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
        modelo_seleccionado = st.selectbox(
            "Modelo de ML", ["Random Forest", "KNN", "SVM", "Regresión lineal"]
        )
        enviar = st.form_submit_button("Estimar gravedad registrada")

    if enviar:
        nombres_archivo = {
            "Random Forest": "random_forest.joblib",
            "KNN": "knn.joblib",
            "SVM": "svm.joblib",
            "Regresión lineal": "regresión_lineal.joblib",
        }
        archivo_modelo = nombres_archivo[modelo_seleccionado]
        modelo_cargado = cargar_modelo(archivo_modelo)
        if modelo_cargado is None and modelo_seleccionado == "Regresión lineal":
            modelo_cargado = cargar_modelo("regresion_lineal.joblib")

        if modelo_cargado is None:
            st.error("No se encontró el modelo entrenado. Ejecuta primero: python src/train_models.py")
        else:
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
            valor = float(np.asarray(modelo_cargado.predict(entrada)).reshape(-1)[0])
            clase = int(valor >= 0.5) if modelo_seleccionado == "Regresión lineal" else int(valor)
            st.subheader(f"Resultado: {ETIQUETAS[clase]}")
            if modelo_seleccionado == "Regresión lineal":
                st.write(
                    f"Salida continua del modelo: {valor:.3f} (se clasifica con umbral 0,5; no es una probabilidad calibrada)."
                )
            elif hasattr(modelo_cargado, "decision_function"):
                st.write(
                    "La clasificación se basa en la función de decisión del SVM; no representa una probabilidad calibrada."
                )
            elif hasattr(modelo_cargado, "predict_proba"):
                probabilidades = modelo_cargado.predict_proba(entrada)[0]
                st.write(
                    f"Salida del modelo para CON HERIDOS: {probabilidades[0]:.1%}; para CON MUERTOS: {probabilidades[1]:.1%}."
                )
            st.caption("Resultado experimental para fines académicos; no usar para decisiones operativas o de seguridad reales.")

with pestanas[3]:
    st.subheader("Alcance y limitaciones")
    st.markdown("""
    - La fuente contiene vehículos involucrados en accidentes registrados, no todos los vehículos que circulan.
    - El modelo aprende asociaciones estadísticas; no demuestra que una marca, edad, ubicación u otra variable cause una gravedad.
    - La salida no equivale a la probabilidad real de sufrir un accidente.
    - La variable objetivo tiene clases desbalanceadas; por eso se informan métricas por clase y matrices de confusión.
    - La fuente no contiene un identificador único de accidente. Los registros idénticos se excluyen del conjunto usado para modelar para reducir fuga de información, pero esta decisión debe reconocerse como limitación.
    - La regresión lineal se incluye por requerimiento académico y se convierte en clase con umbral 0,5; no es un clasificador probabilístico estándar.
    - El SVM implementado es LinearSVC, escogido por escalabilidad frente a una SVM de kernel no lineal sobre cientos de miles de registros.
    """)

    st.subheader("Recomendaciones de prevención vial basadas en los datos")
    st.markdown("""
    - **Enfoque en motociclistas (actores viales vulnerables):** Dado que las motocicletas representan el 49,66 % de los registros analizados, se sugiere priorizar campañas de conducción segura, uso de casco reglamentario y mantenimiento preventivo del sistema de frenos.
    - **Priorización territorial de controles:** Focalizar planes de gestión de velocidad y señalización en los departamentos y municipios con mayor volumen de eventos registrados.
    - **Control técnico-mecánico por antigüedad:** Considerando una edad vehicular media de 11,76 años en el conjunto depurado, se recomienda fortalecer el seguimiento al estado mecánico del parque automotor con más de 10 años de servicio.
    """)