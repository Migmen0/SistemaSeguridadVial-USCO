"""Funciones documentadas para cargar, limpiar y preparar los datos de tránsito."""
from pathlib import Path
import pandas as pd

# Columnas mínimas esperadas en el CSV oficial.
COLUMNAS_REQUERIDAS = [
    "MARCA_VEHICULO", "MODELO_VEHICULO", "TIPO_VEHICULO", "EDAD_VEHICULO",
    "FECHA_ACCIDENTE", "GRAVEDAD_ACCIDENTE", "DEPARTAMENTO_ACCIDENTE",
    "MUNICIPIO_ACCIDENTE", "AUTORIDAD_DE_TRANSITO",
]

# Variables que el sistema usará como entrada para predecir la gravedad registrada.
COLUMNAS_CATEGORICAS = [
    "MARCA_VEHICULO", "TIPO_VEHICULO", "DEPARTAMENTO_ACCIDENTE",
    "MUNICIPIO_ACCIDENTE", "AUTORIDAD_DE_TRANSITO",
]
COLUMNAS_NUMERICAS = [
    "MODELO_VEHICULO", "EDAD_VEHICULO", "ANIO_ACCIDENTE", "MES_ACCIDENTE",
]
COLUMNAS_ENTRADA = COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS
OBJETIVO = "GRAVEDAD_BINARIA"


def cargar_csv(ruta: str | Path) -> pd.DataFrame:
    """Carga el archivo CSV y verifica que contenga las nueve columnas esperadas.

    Args:
        ruta: Ruta al archivo CSV original.

    Returns:
        DataFrame leído desde el archivo.

    Raises:
        ValueError: Si faltan columnas necesarias para el proyecto.
    """
    df = pd.read_csv(ruta, low_memory=False)
    faltantes = sorted(set(COLUMNAS_REQUERIDAS) - set(df.columns))
    if faltantes:
        raise ValueError(f"El CSV no contiene las columnas requeridas: {faltantes}")
    return df


def limpiar_datos(df_original: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpia y transforma los datos sin modificar el archivo original.

    Decisiones metodológicas:
    - Se eliminan espacios externos en campos de texto.
    - Se convierte modelo y edad a valores numéricos.
    - Las edades negativas o superiores a 100 se consideran anómalas y se imputan
      posteriormente durante el entrenamiento; se registran como valores nulos.
    - FECHA_ACCIDENTE se interpreta como MM/AAAA y se divide en año y mes.
    - Se codifica CON HERIDOS como 0 y CON MUERTOS como 1.
    - Se eliminan filas completamente idénticas en la tabla de modelado para
      reducir el riesgo de fuga entre entrenamiento y prueba. Como no hay ID
      único de accidente, esta decisión se documenta como limitación, no como
      prueba de que cada repetición sea un error en la fuente.
    - Se conservan solo las dos categorías de gravedad documentadas.

    Returns:
        Tupla con DataFrame limpio y diccionario de auditoría.
    """
    df = df_original.copy()
    filas_originales = len(df)
    duplicados_exactos = int(df.duplicated().sum())
    faltantes_edad_original = int(df["EDAD_VEHICULO"].isna().sum())

    # Normaliza espacios en las columnas de texto, manteniendo mayúsculas originales.
    for columna in df.select_dtypes(include="object").columns:
        df[columna] = df[columna].astype("string").str.strip()
        df[columna] = df[columna].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Conversión numérica segura: los valores no convertibles quedan como NaN.
    df["MODELO_VEHICULO"] = pd.to_numeric(df["MODELO_VEHICULO"], errors="coerce")
    df["EDAD_VEHICULO"] = pd.to_numeric(df["EDAD_VEHICULO"], errors="coerce")

    # La edad no puede ser negativa; valores mayores de 100 se marcan como anómalos.
    edades_invalidas = df["EDAD_VEHICULO"].lt(0) | df["EDAD_VEHICULO"].gt(100)
    cantidad_edades_invalidas = int(edades_invalidas.sum())
    df.loc[edades_invalidas, "EDAD_VEHICULO"] = pd.NA

    # La fecha oficial está expresada como mes/año, por ejemplo 09/2026.
    fecha = pd.to_datetime(df["FECHA_ACCIDENTE"], format="%m/%Y", errors="coerce")
    df["ANIO_ACCIDENTE"] = fecha.dt.year
    df["MES_ACCIDENTE"] = fecha.dt.month

    # Etiqueta binaria: 0 = accidente registrado con heridos; 1 = con muertos.
    gravedad = df["GRAVEDAD_ACCIDENTE"].str.upper().str.strip()
    df[OBJETIVO] = gravedad.map({"CON HERIDOS": 0, "CON MUERTOS": 1})
    filas_gravedad_no_reconocida = int(df[OBJETIVO].isna().sum())
    df = df.loc[df[OBJETIVO].notna()].copy()

    # Elimina duplicados exactos solo en la tabla de modelado.
    filas_antes_deduplicar = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicados_eliminados = filas_antes_deduplicar - len(df)

    # Categorías faltantes se dejan como NaN; el pipeline las imputa durante el ajuste.
    # El objetivo se mantiene como entero para las métricas de clasificación.
    df[OBJETIVO] = df[OBJETIVO].astype("int8")

    auditoria = {
        "filas_originales": filas_originales,
        "columnas_originales": len(df_original.columns),
        "duplicados_exactos_detectados_en_original": duplicados_exactos,
        "duplicados_exactos_eliminados_para_modelado": int(duplicados_eliminados),
        "faltantes_edad_en_original": faltantes_edad_original,
        "edades_invalidas_marcadas_como_faltantes": cantidad_edades_invalidas,
        "filas_con_gravedad_no_reconocida_eliminadas": filas_gravedad_no_reconocida,
        "filas_finales_modelado": len(df),
        "periodo_inicial": str(fecha.min().strftime("%m/%Y")) if fecha.notna().any() else None,
        "periodo_final": str(fecha.max().strftime("%m/%Y")) if fecha.notna().any() else None,
        "distribucion_gravedad": {
            "CON HERIDOS (0)": int((df[OBJETIVO] == 0).sum()),
            "CON MUERTOS (1)": int((df[OBJETIVO] == 1).sum()),
        },
        "advertencia": (
            "El archivo no tiene un identificador único de accidente. La deduplicación "
            "de filas idénticas reduce el riesgo de fuga, pero no prueba que cada fila "
            "repetida sea un error en la fuente."
        ),
    }
    return df, auditoria
