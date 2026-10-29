"""Entrena y compara KNN, regresión lineal, SVM lineal y Random Forest.

Para que el entrenamiento sea viable en un computador personal, se utiliza una
muestra estratificada de hasta 60.000 filas deduplicadas. El CSV limpio completo
se conserva para análisis y estadísticas. La semilla fija hace reproducible la
muestra y la partición. Las transformaciones se ajustan solo con entrenamiento.
"""
from pathlib import Path
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix,
                             classification_report, mean_absolute_error, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from data_utils import (cargar_csv, limpiar_datos, COLUMNAS_CATEGORICAS,
                        COLUMNAS_NUMERICAS, COLUMNAS_ENTRADA, OBJETIVO)

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO_CSV = RAIZ / "data" / "raw" / "VEHICULOS_INVOLUCRADOS_EN_UN_ACCIDENTE_DE_TRANSITO_LEY_2251-2022_20261027.csv"
DIR_MODELOS = RAIZ / "models"
DIR_REPORTES = RAIZ / "reports"
MAX_FILAS_ENTRENAMIENTO = 20_000
SEMILLA = 42


def crear_preprocesador() -> ColumnTransformer:
    """Construye el tratamiento de variables numéricas y categóricas.

    Los faltantes numéricos se reemplazan por la mediana; los categóricos, por
    la categoría más frecuente. Las categorías se convierten en indicadores.
    La escala numérica se estandariza para que KNN y SVM no estén dominados por
    variables con magnitudes grandes. Se usa un preprocesador dentro de cada
    Pipeline para evitar fuga de información entre entrenamiento y prueba.
    """
    tratamiento_numerico = Pipeline(steps=[
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    tratamiento_categorico = Pipeline(steps=[
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    return ColumnTransformer(transformers=[
        ("numericas", tratamiento_numerico, COLUMNAS_NUMERICAS),
        ("categoricas", tratamiento_categorico, COLUMNAS_CATEGORICAS),
    ])


def obtener_estimadores() -> dict:
    """Devuelve los cuatro modelos solicitados con parámetros explícitos."""
    return {
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
        # Se usa LinearSVC: es una implementación SVM lineal más escalable que SVC RBF.
        "SVM": LinearSVC(class_weight="balanced", random_state=SEMILLA, max_iter=3000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=18, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=SEMILLA, n_jobs=-1,
        ),
        # Se ajusta a la etiqueta 0/1 y se umbraliza en 0.5 para obtener clase.
        "Regresión lineal": LinearRegression(n_jobs=-1),
    }


def calcular_metricas(nombre: str, modelo: Pipeline, x_prueba: pd.DataFrame,
                       y_prueba: pd.Series) -> tuple[dict, dict]:
    """Calcula métricas de clasificación y guarda detalles de evaluación.

    La regresión lineal no es un clasificador probabilístico; sus predicciones
    continuas se convierten en clase con umbral 0.5 y se reportan MAE/RMSE aparte.
    Para los otros modelos se utiliza predict; cuando hay score se calcula AUC.
    """
    predicciones_continuas = np.asarray(modelo.predict(x_prueba)).reshape(-1)
    predicciones = (predicciones_continuas >= 0.5).astype(int) if nombre == "Regresión lineal" else predicciones_continuas.astype(int)
    y_true = y_prueba.to_numpy(dtype=int)
    matriz = confusion_matrix(y_true, predicciones, labels=[0, 1])
    fila = {
        "modelo": nombre,
        "accuracy": float(accuracy_score(y_true, predicciones)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicciones)),
        "precision_clase_muertos": float(precision_score(y_true, predicciones, pos_label=1, zero_division=0)),
        "recall_clase_muertos": float(recall_score(y_true, predicciones, pos_label=1, zero_division=0)),
        "f1_clase_muertos": float(f1_score(y_true, predicciones, pos_label=1, zero_division=0)),
        "precision_macro": float(precision_score(y_true, predicciones, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predicciones, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predicciones, average="macro", zero_division=0)),
    }
    try:
        if nombre == "Regresión lineal":
            fila["mae_regresion"] = float(mean_absolute_error(y_true, predicciones_continuas))
            fila["rmse_regresion"] = float(np.sqrt(mean_squared_error(y_true, predicciones_continuas)))
        elif hasattr(modelo, "decision_function"):
            fila["roc_auc"] = float(roc_auc_score(y_true, modelo.decision_function(x_prueba)))
        elif hasattr(modelo, "predict_proba"):
            fila["roc_auc"] = float(roc_auc_score(y_true, modelo.predict_proba(x_prueba)[:, 1]))
    except (ValueError, AttributeError):
        fila["roc_auc"] = None

    detalle = {
        "modelo": nombre,
        "matriz_confusion": matriz.tolist(),
        "orden_matriz": ["real CON HERIDOS / CON MUERTOS", "predicción CON HERIDOS / CON MUERTOS"],
        "reporte_clasificacion": classification_report(
            y_true, predicciones, labels=[0, 1], target_names=["CON HERIDOS", "CON MUERTOS"],
            output_dict=True, zero_division=0,
        ),
    }
    return fila, detalle


def main() -> None:
    """Prepara muestra, entrena los modelos y guarda artefactos y métricas."""
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    DIR_REPORTES.mkdir(parents=True, exist_ok=True)

    # Limpia desde el archivo original para que la ejecución sea reproducible.
    original = cargar_csv(ARCHIVO_CSV)
    datos, auditoria = limpiar_datos(original)
    datos.to_csv(RAIZ / "data" / "processed" / "vehiculos_limpios.csv", index=False, encoding="utf-8-sig")
    (DIR_REPORTES / "auditoria_limpieza.json").write_text(json.dumps(auditoria, ensure_ascii=False, indent=2), encoding="utf-8")

    # Solo se usan las columnas definidas antes de observar los resultados del test.
    datos = datos.dropna(subset=[OBJETIVO]).copy()
    if len(datos) > MAX_FILAS_ENTRENAMIENTO:
        # Muestreo estratificado: conserva aproximadamente la proporción de ambas clases.
        muestra, _ = train_test_split(
            datos, train_size=MAX_FILAS_ENTRENAMIENTO, random_state=SEMILLA,
            stratify=datos[OBJETIVO],
        )
    else:
        muestra = datos

    x = muestra[COLUMNAS_ENTRADA].copy()
    y = muestra[OBJETIVO].astype(int).copy()
    x_entrena, x_prueba, y_entrena, y_prueba = train_test_split(
        x, y, test_size=0.20, random_state=SEMILLA, stratify=y,
    )
    print(f"Filas limpias: {len(datos):,}")
    print(f"Muestra usada para modelado: {len(muestra):,}")
    print(f"Entrenamiento: {len(x_entrena):,}; prueba: {len(x_prueba):,}")
    print("Distribución en prueba:", y_prueba.value_counts(normalize=True).round(4).to_dict())

    resultados, detalles = [], []
    for nombre, estimador in obtener_estimadores().items():
        print(f"\nEntrenando {nombre}...")
        inicio = time.time()
        pipeline = Pipeline(steps=[("preprocesamiento", crear_preprocesador()), ("modelo", estimador)])
        # Regresión lineal y clasificadores se ajustan sobre la misma etiqueta 0/1.
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            pipeline.fit(x_entrena, y_entrena)
        segundos = time.time() - inicio
        fila, detalle = calcular_metricas(nombre, pipeline, x_prueba, y_prueba)
        fila["segundos_entrenamiento"] = round(segundos, 2)
        resultados.append(fila)
        detalles.append(detalle)
        joblib.dump(pipeline, DIR_MODELOS / f"{nombre.lower().replace(' ', '_')}.joblib", compress=3)
        print(f"{nombre}: F1 muertos={fila['f1_clase_muertos']:.4f}; tiempo={segundos:.1f}s")

    comparacion = pd.DataFrame(resultados).sort_values("f1_macro", ascending=False)
    comparacion.to_csv(DIR_REPORTES / "comparacion_modelos.csv", index=False, encoding="utf-8-sig")
    (DIR_REPORTES / "detalle_evaluacion.json").write_text(json.dumps(detalles, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "columnas_entrada": COLUMNAS_ENTRADA,
        "columnas_categoricas": COLUMNAS_CATEGORICAS,
        "columnas_numericas": COLUMNAS_NUMERICAS,
        "objetivo": OBJETIVO,
        "codificacion_objetivo": {"0": "CON HERIDOS", "1": "CON MUERTOS"},
        "max_filas_modelado": MAX_FILAS_ENTRENAMIENTO,
        "semilla": SEMILLA,
        "nota": "Las métricas describen gravedad registrada entre vehículos involucrados, no riesgo causal ni probabilidad de accidente para cualquier vehículo en circulación.",
        "limitacion_regresion_lineal": "Se umbraliza en 0.5; su salida no es una probabilidad calibrada.",
        "limitacion_svm": "Se usa LinearSVC por escalabilidad en una tabla de cientos de miles de registros.",
    }
    (DIR_MODELOS / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nCOMPARACIÓN FINAL")
    print(comparacion.to_string(index=False))
    print(f"\nResultados guardados en: {DIR_REPORTES}")
    print(f"Modelos guardados en: {DIR_MODELOS}")


if __name__ == "__main__":
    main()
