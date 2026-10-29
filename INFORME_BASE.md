# Sistema inteligente para la predicción y prevención de riesgos en la seguridad vial urbana mediante análisis de datos

**Autor:** Miguel Esteban Mendoza Calderon  
**Área:** Ingeniería de software / análisis de datos  
**Fecha de actualización de este prototipo:** 27 de octubre de 2026

> Este documento es una base editable. Antes de entregarlo, agrega los nombres de los integrantes, asignatura, docente, institución, portada y referencias exigidas por el profesor. Las cifras de resultados corresponden a la ejecución de referencia incluida en este proyecto y deben volver a generarse si cambias la muestra, el código o la versión del archivo.

## 1. Resumen

Este proyecto desarrolla un prototipo de análisis de datos para estudiar la gravedad registrada en accidentes de tránsito en Colombia. Se utiliza un conjunto de datos abierto que contiene información de vehículos involucrados en accidentes con heridos o muertos, incluyendo marca, modelo, tipo de vehículo, edad, fecha mensual, departamento, municipio y autoridad de tránsito. Se implementó una preparación reproducible de los datos y se compararon cuatro enfoques de aprendizaje automático: K-Nearest Neighbors (KNN), regresión lineal, Support Vector Machine lineal (SVM) y Random Forest. La variable objetivo se definió como gravedad registrada, con `CON HERIDOS = 0` y `CON MUERTOS = 1`. Para comparar los modelos se calcularon accuracy, balanced accuracy, precisión, recall y F1, con especial atención a la clase minoritaria `CON MUERTOS`. En la ejecución de referencia, Random Forest obtuvo el mayor F1 para la clase `CON MUERTOS` (0,308) y un recall de 0,609; sin embargo, su precisión fue 0,206, lo que indica una cantidad importante de falsos positivos. Los resultados son exploratorios: permiten estudiar patrones dentro de registros de vehículos ya involucrados en accidentes, pero no calculan la probabilidad real de que cualquier vehículo tenga un accidente ni demuestran causalidad.

## 2. Planteamiento del problema

Los registros de siniestros viales contienen información que puede analizarse para describir patrones de gravedad y apoyar la exploración de factores asociados a los resultados reportados. No obstante, los datos presentan retos como categorías de texto, valores faltantes o atípicos, filas completamente idénticas y un desbalance entre los registros clasificados como `CON HERIDOS` y `CON MUERTOS`. Además, el conjunto no contiene un identificador único de accidente que permita distinguir con certeza todas las observaciones repetidas. Por ello, se requiere un proceso de limpieza documentado y una evaluación que no dependa únicamente de la exactitud global.

## 3. Objetivo general

Desarrollar un prototipo de análisis de datos que prepare información de vehículos involucrados en accidentes de tránsito y compare modelos de aprendizaje automático para estimar la gravedad registrada como `CON HERIDOS` o `CON MUERTOS`, presentando sus resultados y limitaciones mediante una aplicación interactiva.

## 4. Objetivos específicos

1. Inspeccionar y limpiar el archivo CSV, documentando valores faltantes, edades anómalas y registros idénticos.
2. Transformar las variables para facilitar el análisis y el entrenamiento de modelos.
3. Implementar KNN, regresión lineal, SVM lineal y Random Forest bajo una partición de entrenamiento y prueba reproducible.
4. Comparar los modelos con métricas de clasificación sensibles al desbalance de clases.
5. Construir una aplicación de demostración para consultar estadísticas y probar los modelos entrenados.
6. Explicar las limitaciones para evitar interpretar las predicciones como causalidad o como probabilidad real de accidente.

## 5. Fuente y descripción de los datos

Se trabajó con el archivo `VEHICULOS_INVOLUCRADOS_EN_UN_ACCIDENTE_DE_TRANSITO_LEY_2251-2022_20261027.csv`, descargado por el usuario desde el portal de datos abiertos. El archivo analizado contiene **450.335 filas y 9 columnas**, con fechas mensuales desde **enero de 2022 hasta septiembre de 2026**. La versión del archivo debe quedar identificada en el informe porque la fuente puede actualizarse y sus cifras pueden cambiar.

Las variables originales son:

- `MARCA_VEHICULO`: marca del vehículo.
- `MODELO_VEHICULO`: año modelo reportado.
- `TIPO_VEHICULO`: clase de vehículo.
- `EDAD_VEHICULO`: edad del vehículo en años.
- `FECHA_ACCIDENTE`: mes y año del registro (`MM/AAAA`).
- `GRAVEDAD_ACCIDENTE`: gravedad registrada (`CON HERIDOS` o `CON MUERTOS`).
- `DEPARTAMENTO_ACCIDENTE`: departamento del registro.
- `MUNICIPIO_ACCIDENTE`: municipio del registro.
- `AUTORIDAD_DE_TRANSITO`: autoridad reportada.

En el archivo original, `CON HERIDOS` aparece en 426.436 filas (94,69 %) y `CON MUERTOS` en 23.899 filas (5,31 %). Las motocicletas son 255.266 filas (56,68 %). Estas cifras describen el archivo original, antes de retirar filas idénticas; no son tasas de riesgo porque no se dispone del total de vehículos expuestos ni de todos los desplazamientos.

## 6. Limpieza y preparación

La inspección inicial encontró 20 valores faltantes en `EDAD_VEHICULO` y 159.399 filas completamente idénticas. También se detectaron 63 valores de edad superiores a 100 años. El proceso de preparación implementado realiza las siguientes operaciones:

1. Conserva intacto el CSV original y trabaja sobre una copia.
2. Elimina espacios al inicio y al final de los campos de texto.
3. Convierte `MODELO_VEHICULO` y `EDAD_VEHICULO` a valores numéricos.
4. Marca como faltantes las edades negativas o superiores a 100 años. La imputación posterior se realiza dentro del pipeline y se aprende únicamente a partir de los datos de entrenamiento.
5. Convierte `FECHA_ACCIDENTE` en dos variables: `ANIO_ACCIDENTE` y `MES_ACCIDENTE`.
6. Codifica la variable objetivo como `CON HERIDOS = 0` y `CON MUERTOS = 1`.
7. Retira filas exactamente duplicadas de la tabla destinada al modelado para reducir el riesgo de que observaciones idénticas aparezcan en entrenamiento y prueba. Como no existe un identificador único de accidente, no se puede asegurar que todas las filas repetidas sean errores. Esta decisión es una limitación y debe interpretarse con cautela.
8. Aplica imputación y codificación dentro de cada pipeline de modelo. Las categorías se transforman con One-Hot Encoding y las variables numéricas se escalan.

Después de retirar las filas idénticas, quedaron **290.936 filas** para modelado: 267.633 con `CON HERIDOS` (92,00 %) y 23.303 con `CON MUERTOS` (8,00 %). La deduplicación cambia la distribución de clases, por lo que se informan separadamente las cifras del archivo original y las de la tabla preparada.

## 7. Formulación matemática

Sea cada observación representada por un vector de características:

\[
\mathbf{x}_i = (marca_i, modelo_i, tipo_i, edad_i, año_i, mes_i, departamento_i, municipio_i, autoridad_i)
\]

La variable objetivo binaria es:

\[
y_i = \begin{cases}0, & \text{si la gravedad registrada es CON HERIDOS}\\1, & \text{si la gravedad registrada es CON MUERTOS}\end{cases}
\]

El objetivo experimental es aprender una función \(f(\mathbf{x})\) que estime la etiqueta registrada:

\[
\hat{y}_i = f(\mathbf{x}_i), \qquad \hat{y}_i \in \{0,1\}
\]

Esta formulación describe una clasificación de gravedad dentro de los registros disponibles; no estima la probabilidad de que un vehículo cualquiera tenga un accidente.

### 7.1 K-Nearest Neighbors (KNN)

KNN clasifica una observación según las etiquetas de sus vecinos más cercanos en el espacio de características transformadas. Una distancia euclidiana conceptual es:

\[
d(\mathbf{x},\mathbf{x}_j)=\sqrt{\sum_{k=1}^{p}(x_k-x_{jk})^2}
\]

En el código se utiliza `n_neighbors=7` y ponderación por distancia. El escalado de variables numéricas evita que sus magnitudes dominen la distancia; las categorías se convierten en indicadores.

### 7.2 Regresión lineal

La regresión lineal ajusta una salida continua:

\[
\hat{y}=\beta_0+\sum_{j=1}^{p}\beta_jx_j
\]

Como se incluye por requisito académico, se entrena usando la etiqueta 0/1 y se convierte su salida en clase con el umbral 0,5: si \(\hat{y}\geq 0,5\), se asigna clase 1; de lo contrario, clase 0. **Esta técnica no es un clasificador probabilístico estándar** y su salida no debe interpretarse como probabilidad calibrada. Los resultados muestran que su accuracy puede ser alto y, al mismo tiempo, su recall para `CON MUERTOS` puede ser bajo.

### 7.3 Support Vector Machine (SVM)

SVM busca una frontera que separe las clases maximizando el margen entre observaciones de diferentes etiquetas. Para facilitar el entrenamiento sobre una tabla grande, se utiliza `LinearSVC`, una variante lineal. Se ponderan las clases para reducir el efecto del desbalance. Su función de decisión no se interpreta como una probabilidad calibrada.

### 7.4 Random Forest

Random Forest combina múltiples árboles de decisión. Cada árbol produce una clasificación y el conjunto combina sus resultados para obtener la clase final. En el código se usa un bosque de 100 árboles, profundidad máxima 18 y ponderación balanceada por submuestra. Esta configuración se eligió como compromiso entre capacidad de aprendizaje y tiempo de ejecución.

## 8. Diseño experimental

Para que el entrenamiento sea viable en un computador personal, se seleccionó una muestra aleatoria estratificada de hasta 20.000 filas de la tabla deduplicada. Se utilizó una semilla fija (`42`) para favorecer la reproducibilidad. La muestra se dividió en 80 % para entrenamiento (16.000 filas) y 20 % para prueba (4.000 filas), preservando aproximadamente la distribución de las clases. Los cuatro modelos se evaluaron sobre la misma partición. El preprocesamiento se ajustó dentro de los pipelines usando únicamente el conjunto de entrenamiento para reducir la fuga de información.

## 9. Métricas de evaluación

- **Accuracy:** proporción total de predicciones correctas.
- **Balanced accuracy:** promedio del recall de cada clase.
- **Precisión para `CON MUERTOS`:** entre los registros predichos como muertos, cuántos realmente pertenecían a esa clase.
- **Recall para `CON MUERTOS`:** de todos los registros realmente `CON MUERTOS`, qué proporción identificó el modelo.
- **F1 para `CON MUERTOS`:** media armónica de precisión y recall de esa clase.
- **F1 macro:** promedio del F1 de ambas clases, con el mismo peso por clase.
- **AUC ROC:** capacidad de ordenar las dos clases a partir de un score cuando está disponible.
- **Matriz de confusión:** muestra aciertos y errores por clase.
- **MAE y RMSE:** se reportan para la salida continua de la regresión lineal.

Debido al desbalance, un modelo que predijera casi siempre `CON HERIDOS` podría obtener accuracy alto sin detectar adecuadamente `CON MUERTOS`. Por ello, la interpretación debe considerar principalmente las métricas por clase y la matriz de confusión.

## 10. Resultados de la ejecución de referencia

La prueba incluyó 4.000 filas, con aproximadamente 92 % de la clase `CON HERIDOS` y 8 % de `CON MUERTOS` en la muestra deduplicada. La tabla presenta métricas obtenidas al ejecutar `src/train_models.py`.

| Modelo | Accuracy | Balanced accuracy | Precisión CON MUERTOS | Recall CON MUERTOS | F1 CON MUERTOS | F1 macro | AUC ROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| KNN | 0,9145 | 0,5213 | 0,3036 | 0,0531 | 0,0904 | 0,5228 | 0,6854 |
| Regresión lineal | 0,9165 | 0,5224 | 0,3542 | 0,0531 | 0,0924 | 0,5243 | No aplica |
| SVM lineal | 0,7560 | 0,6734 | 0,1797 | 0,5750 | 0,2738 | 0,5636 | 0,7319 |
| Random Forest | 0,7810 | 0,7026 | 0,2061 | 0,6094 | 0,3081 | 0,5890 | 0,7788 |

### 10.1 Análisis de los resultados

Random Forest obtuvo el mayor F1 macro (0,5890), la mayor balanced accuracy (0,7026) y el mayor recall para `CON MUERTOS` (0,6094). Esto significa que identificó una proporción mayor de los casos de la clase minoritaria que los otros modelos evaluados en esta ejecución. Sin embargo, su precisión para `CON MUERTOS` fue 0,2061; por tanto, muchas predicciones de esa clase fueron falsos positivos. El modelo no debe considerarse listo para uso real, pero es el candidato más prometedor entre los cuatro para continuar experimentando con esta configuración.

KNN y regresión lineal alcanzaron accuracy superior a 0,91, pero su recall de `CON MUERTOS` fue apenas 0,0531. En términos prácticos, identificaron cerca del 5,3 % de los registros de esa clase en la prueba. Este resultado ilustra por qué la exactitud global no es suficiente para evaluar un conjunto desbalanceado. SVM mejoró el recall hasta 0,5750, aunque con precisión baja. Random Forest y SVM muestran un intercambio entre detectar más casos de `CON MUERTOS` y generar más falsas alarmas.

Las métricas corresponden a una muestra de 20.000 filas y a una configuración concreta de hiperparámetros. No constituyen una validación independiente en datos futuros ni garantizan generalización a otros municipios o periodos. Se recomienda validar en una partición temporal y ajustar umbrales e hiperparámetros antes de cualquier uso fuera del contexto académico.

## 11. Aplicación desarrollada

La aplicación de Streamlit incluye cuatro secciones:

1. **Resumen de datos:** presenta el total de registros, departamentos, motocicletas y distribución de gravedad.
2. **Comparación de modelos:** muestra las métricas calculadas en el conjunto de prueba.
3. **Demostración de predicción:** permite ingresar características y seleccionar uno de los modelos entrenados para obtener una estimación experimental de la gravedad registrada.
4. **Limitaciones:** explica el alcance del modelo y advierte contra la interpretación causal o probabilística indebida.

La aplicación está diseñada como prototipo académico y no debe emplearse para decisiones reales de seguridad vial.

## 12. Limitaciones

1. Los registros describen vehículos involucrados en accidentes; no hay un denominador que represente todos los vehículos o viajes expuestos al riesgo.
2. Las asociaciones estadísticas no demuestran causalidad.
3. No existe un identificador único de accidente en las columnas utilizadas, por lo que no se puede confirmar si todas las filas idénticas son duplicados erróneos.
4. La deduplicación modifica la distribución de las clases.
5. Se entrena con una muestra estratificada de 20.000 registros por motivos de tiempo y recursos; los resultados podrían cambiar al variar la muestra.
6. La clase `CON MUERTOS` es minoritaria y los falsos positivos siguen siendo importantes.
7. La regresión lineal no es un clasificador probabilístico estándar y el umbral 0,5 es una decisión práctica para comparación académica.
8. `LinearSVC` es lineal; no captura necesariamente relaciones no lineales complejas.
9. Las fechas de 2026 corresponden a los registros disponibles en el archivo descargado y pueden estar incompletas.

## 13. Conclusiones

Se construyó un flujo reproducible para limpiar los datos, transformar las variables, entrenar cuatro modelos y comparar su rendimiento con métricas sensibles al desbalance de clases. La evaluación de referencia mostró que Random Forest obtuvo el mejor balance global entre las métricas seleccionadas y el mayor recall para `CON MUERTOS`, mientras que KNN y regresión lineal presentaron accuracy alto, pero una capacidad muy limitada para identificar la clase minoritaria. Por tanto, la selección del modelo debe basarse en el objetivo de análisis y no solo en la exactitud.

El prototipo demuestra cómo integrar análisis de datos, aprendizaje automático y una interfaz de consulta. No obstante, los resultados deben considerarse exploratorios. La fuente no permite calcular la probabilidad de accidente de cualquier vehículo ni demostrar que las características utilizadas causen una gravedad determinada. Para fortalecer el trabajo se recomienda evaluar con partición temporal, revisar la definición de duplicados con el proveedor de los datos, ajustar hiperparámetros, explorar técnicas de balanceo solo dentro del entrenamiento y validar el sistema con especialistas en seguridad vial.

## 14. Referencia de la fuente de datos

Datos Abiertos de Colombia. *Vehículos involucrados en un accidente de tránsito Ley 2251-2022*. Portal oficial de datos abiertos. URL: https://www.datos.gov.co/Transporte/VEHICULOS-INVOLUCRADOS-EN-UN-ACCIDENTE-DE-TRANSITO/6jmc-vaxk

> Verifica la ficha del portal y registra la fecha exacta de descarga en la versión final del informe.
