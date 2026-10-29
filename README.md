# Sistema inteligente para la predicción y prevención de riesgos en la seguridad vial urbana mediante análisis de datos

Proyecto académico en Python: limpieza, análisis exploratorio, entrenamiento/comparación de KNN, regresión lineal, SVM lineal y Random Forest, y una demostración con Streamlit.

## Alcance científico

La base contiene vehículos involucrados en accidentes de tránsito registrados. El objetivo implementado es estimar la gravedad registrada (`CON HERIDOS` frente a `CON MUERTOS`) entre esos registros. **No** se debe afirmar que el modelo calcula la probabilidad de que cualquier vehículo tenga un accidente, ni que demuestra causalidad.

## 1. Requisitos

- Python 3.11 recomendado.
- Visual Studio Code.
- La carpeta `data/raw/` debe contener el CSV original llamado exactamente:
  `VEHICULOS_INVOLUCRADOS_EN_UN_ACCIDENTE_DE_TRANSITO_LEY_2251-2022_20261027.csv`.
- Instalar las dependencias indicadas en `requirements.txt`.

## 2. Abrir el proyecto en VS Code

1. Descomprime el ZIP en una carpeta fácil de encontrar, por ejemplo `C:\Miguel\SistemaSeguridadVial`.
2. Abre VS Code y selecciona **File > Open Folder**.
3. Abre una terminal integrada: **Terminal > New Terminal**.
4. Crea y activa un entorno virtual:

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si `py -3.11` no funciona, instala Python 3.11 y marca la opción para agregarlo al PATH. Si PowerShell bloquea la activación, se puede ejecutar `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` en esa terminal y volver a activar.

## 3. Ejecutar limpieza y entrenamiento

Desde la raíz del proyecto:

```powershell
python src/train_models.py
```

Este script también realiza la limpieza y guarda `data/processed/vehiculos_limpios.csv`. Al finalizar crea:

- `models/knn.joblib`
- `models/regresion_lineal.joblib`
- `models/svm.joblib`
- `models/random_forest.joblib`
- `reports/auditoria_limpieza.json`
- `reports/comparacion_modelos.csv`
- `reports/detalle_evaluacion.json`

El script utiliza una muestra estratificada de hasta 20.000 filas para hacer viable el entrenamiento en un computador personal. El conjunto limpio completo queda guardado. La muestra se usa para comparar modelos de forma consistente.

## 4. Abrir la aplicación

Después de que el entrenamiento termine correctamente:

```powershell
python -m streamlit run app.py
```

Streamlit abrirá la aplicación en el navegador. Usa las pestañas para consultar resumen, comparación de modelos y formulario de demostración.

## 5. Cómo interpretar las métricas

- **Accuracy:** fracción de predicciones correctas; puede ser engañosa con clases desbalanceadas.
- **Balanced accuracy:** promedio del recall de cada clase.
- **Precision de CON MUERTOS:** proporción de predicciones de muertos que fueron correctas.
- **Recall de CON MUERTOS:** proporción de registros CON MUERTOS identificados por el modelo.
- **F1 de CON MUERTOS:** equilibrio entre precisión y recall de esa clase.
- **F1 macro:** F1 promedio entre las dos clases, con el mismo peso para cada clase.
- **Matriz de confusión:** compara clase real y predicha.

No se debe elegir el mejor modelo por accuracy solamente. Se debe discutir el recall y F1 de `CON MUERTOS`, además de la matriz de confusión.

## 6. Decisiones de limpieza que deben explicarse en el informe

- Los valores de edad negativos o superiores a 100 se consideran anomalías y quedan como faltantes; el pipeline los imputa con la mediana calculada exclusivamente con los datos de entrenamiento.
- La fecha `MM/AAAA` se divide en año y mes.
- `CON HERIDOS` se codifica como 0 y `CON MUERTOS` como 1.
- Se eliminan filas completamente idénticas para el modelado para reducir fuga entre entrenamiento y prueba. No se puede afirmar que todos los duplicados sean errores porque no hay un ID único de accidente. Esta es una limitación metodológica.
- El conjunto de prueba se separa de forma estratificada para preservar la proporción de las dos clases.
- El preprocesamiento se aprende dentro de cada pipeline usando solo el entrenamiento, evitando fuga de información.
- KNN y SVM se benefician de la escala numérica; las categorías se codifican con One-Hot Encoding.
- Se usa `LinearSVC` como variante escalable de SVM lineal. La regresión lineal se incluye por requerimiento académico y su salida se umbraliza en 0,5; no es una probabilidad calibrada.

## 7. Estructura

```text
SistemaSeguridadVial/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/       # CSV original
│   └── processed/ # CSV limpio generado
├── models/        # modelos entrenados y metadata
├── reports/       # auditoría y métricas
└── src/
    ├── data_utils.py
    ├── clean_dataset.py
    └── train_models.py
```

## 8. Posible texto de metodología para el informe

Se realizó una preparación reproducible de los registros mediante normalización de cadenas, conversión de variables numéricas, revisión de edades atípicas, transformación de la fecha mensual y codificación binaria de la gravedad. Para reducir el riesgo de fuga de información, se retiraron las filas exactamente duplicadas antes de la división para entrenamiento y prueba; esta decisión se reconoce como limitación debido a que la fuente no presenta un identificador único del accidente. Se definió como variable objetivo la gravedad registrada, codificando `CON HERIDOS` como 0 y `CON MUERTOS` como 1. Se utilizó una partición estratificada del 80 % para entrenamiento y 20 % para prueba, con semilla fija. Para hacer viable la experimentación en un equipo personal se empleó una muestra estratificada de hasta 20.000 registros. Se compararon K-Nearest Neighbors (KNN), regresión lineal con umbral de 0,5, Support Vector Machine lineal (LinearSVC) y Random Forest. La evaluación incluye accuracy, balanced accuracy, precisión, recall, F1 por clase, F1 macro y matriz de confusión; para la regresión lineal se reportan también MAE y RMSE de la salida continua. Debido al desbalance de clases, se presta especial atención al desempeño de la clase `CON MUERTOS`. Los resultados se interpretan como asociaciones sobre registros de vehículos ya involucrados en accidentes y no como causalidad ni como probabilidad de accidente para todo vehículo en circulación.
