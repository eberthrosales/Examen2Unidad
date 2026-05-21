# 🧠 ML & Data Science Platform

> **Plataforma profesional de análisis de datos e inteligencia artificial — compatible con cualquier dataset tabular, sin configuración manual.**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Tabla de Contenidos

1. [Descripción General](#-descripción-general)
2. [Características Principales](#-características-principales)
3. [Stack Tecnológico](#️-stack-tecnológico)
4. [Estructura del Proyecto](#-estructura-del-proyecto)
5. [Instalación y Configuración](#-instalación-y-configuración)
6. [Cómo Ejecutar la Aplicación](#-cómo-ejecutar-la-aplicación)
7. [Flujo de Trabajo Paso a Paso](#-flujo-de-trabajo-paso-a-paso)
8. [Módulos del Sistema](#-módulos-del-sistema)
9. [Configuración de Streamlit](#-configuración-de-streamlit)
10. [Errores Conocidos y Soluciones](#-errores-conocidos-y-soluciones)
11. [Capturas de Pantalla](#-capturas-de-pantalla)

---

## 📌 Descripción General

**ML & Data Science Platform** es una aplicación web interactiva construida con **Streamlit** que permite realizar un flujo completo de Data Science y Machine Learning sobre cualquier dataset tabular (CSV o Excel), sin necesidad de escribir código.

La plataforma guía al usuario paso a paso desde la carga de datos hasta la exportación de reportes ejecutivos, cubriendo:

- Análisis Exploratorio de Datos (EDA) automático
- Preprocesamiento con pipeline anti-leakage (Scikit-learn Pipeline)
- Partición Train / Validación / Test
- Clustering no supervisado (K-Means + Jerárquico)
- Clasificación supervisada (Árbol de Decisión + Random Forest)
- Evaluación comparativa con métricas profesionales
- Exportación en PDF, Excel, CSV y JSON

---

## ✨ Características Principales

| Característica | Descripción |
|---|---|
| 🔍 **EDA Automático** | Detecta tipos de variables, distribuciones, correlaciones y outliers automáticamente |
| ⚙️ **Pipeline Inteligente** | Sklearn Pipeline anti-leakage: fit solo en train, transform en val/test |
| 🎯 **Detección de Target** | Sugiere automáticamente variables objetivo candidatas |
| 🔮 **Clustering Completo** | K-Means + Jerárquico con Método del Codo, Silhouette, PCA/t-SNE |
| 🌳 **Clasificación Supervisada** | Árbol de Decisión + Random Forest con validación cruzada |
| 📊 **Evaluación Comparativa** | Radar chart, curvas ROC, matrices de confusión, ranking de modelos |
| 📤 **Exportación Completa** | PDF ejecutivo, Excel multi-hoja, JSON, CSV de predicciones |
| 🎨 **UI Premium** | Tema oscuro con glassmorphism, animaciones y diseño profesional |
| 🔒 **Anti Data Leakage** | El pipeline de transformaciones se ajusta exclusivamente con datos de entrenamiento |

---

## 🛠️ Stack Tecnológico

```
Frontend / UI
├── Streamlit >= 1.32.0        — Framework web interactivo
└── CSS personalizado          — Tema oscuro, glassmorphism, animations

Procesamiento de Datos
├── Pandas >= 2.0.0            — Manipulación de DataFrames
├── NumPy >= 1.24.0            — Computación numérica
└── SciPy >= 1.11.0            — Estadística y clustering jerárquico

Machine Learning
├── Scikit-learn >= 1.4.0      — Pipeline, modelos, métricas, preprocesamiento
└── Joblib >= 1.3.0            — Serialización de modelos

Visualización
├── Plotly >= 5.18.0           — Gráficos interactivos
├── Matplotlib >= 3.7.0        — Visualizaciones estáticas (árbol, dendrograma)
└── Seaborn >= 0.13.0          — Heatmaps y distribuciones

Exportación
├── FPDF2 >= 2.7.6             — Generación de PDFs
├── OpenPyXL >= 3.1.0          — Lectura/escritura Excel
├── XlsxWriter >= 3.1.0        — Excel con formatos avanzados
└── Kaleido >= 0.2.1           — Exportación de gráficos Plotly a imagen
```

---

## 📁 Estructura del Proyecto

```
Appmineria/
│
├── app.py                      # Punto de entrada principal — Router y Sidebar
│
├── requirements.txt            # Dependencias del proyecto
│
├── .streamlit/
│   └── config.toml             # Configuración del tema oscuro y servidor
│
├── assets/
│   └── style.css               # Estilos CSS personalizados (glassmorphism, cards)
│
└── modules/
    ├── __init__.py
    ├── data_loader.py          # Carga de CSV/Excel con detección automática
    ├── eda.py                  # Análisis Exploratorio de Datos completo
    ├── preprocessing.py        # Pipeline de preprocesamiento anti-leakage
    ├── partitioning.py         # Partición Train/Val/Test + Baseline
    ├── clustering.py           # K-Means y Clustering Jerárquico
    ├── classification.py       # Árbol de Decisión y Random Forest
    ├── evaluation.py           # Métricas, ROC, Confusión, Evaluación Test
    ├── interpretation.py       # Generación automática de textos explicativos
    ├── export.py               # Exportación PDF, Excel, CSV, JSON
    └── visualization.py        # Funciones de gráficos reutilizables (Plotly)
```

---

## 🚀 Instalación y Configuración

### Prerrequisitos

- **Python 3.11 o superior** — [Descargar](https://www.python.org/downloads/)
- **pip** (incluido con Python)
- **Git** — [Descargar](https://git-scm.com/)

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/eberthrosales/Examen2Unidad.git
cd Examen2Unidad
```

### Paso 2: Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

> ⚠️ **Nota sobre Kaleido y Plotly**: Si ves una advertencia de incompatibilidad de versiones entre Plotly y Kaleido, la funcionalidad principal de la app seguirá funcionando. Solo la exportación de imágenes estáticas podría verse afectada. Para resolverlo:
> ```bash
> pip install plotly>=6.1.1
> # o bien:
> pip install kaleido==0.2.1
> ```

---

## ▶️ Cómo Ejecutar la Aplicación

```bash
# Desde la raíz del proyecto:
python -m streamlit run app.py

# O con streamlit directamente:
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en:
- **Local:** http://localhost:8501
- **Red:** http://\<tu-ip\>:8501

Para detener la aplicación: `Ctrl + C` en la terminal.

---

## 🗺️ Flujo de Trabajo Paso a Paso

La plataforma sigue un flujo lineal de 8 pasos. La barra lateral muestra el progreso y bloquea secciones que requieren pasos previos.

---

### 📂 Paso 1 — Carga de Datos

**Sección:** `1. Carga de Datos`

1. Haz clic en **"Browse files"** o arrastra tu archivo al área de carga.
2. Formatos soportados:
   - **CSV**: cualquier delimitador (`,` `;` `TAB` `|` `:`) — detección automática
   - **Excel**: `.xlsx` / `.xls` con selección de hoja
3. El sistema automáticamente:
   - Detecta el tipo de cada columna (numérica, categórica, fecha, ID, texto libre)
   - Calcula estadísticas de nulos
   - Sugiere posibles **variables objetivo** para clasificación
4. Se muestra una vista previa y tabla de información de columnas.
5. Haz clic en **"➡️ Ir a Análisis Exploratorio (EDA)"** para continuar.

---

### 🔍 Paso 2 — Exploración (EDA)

**Sección:** `2. Exploración (EDA)`

Contiene múltiples pestañas:

| Pestaña | Contenido |
|---|---|
| 📊 Resumen | Estadísticas descriptivas completas |
| 📈 Distribuciones | Histogramas y boxplots para todas las variables numéricas |
| 🔗 Correlaciones | Heatmap de correlación de Pearson |
| 🎯 Variable Objetivo | Distribución de clases y balance |
| ❌ Valores Nulos | Mapa de calor de nulos y porcentajes |
| 🔍 Outliers | Detección con método IQR |
| 📋 Datos Crudos | Vista completa del DataFrame |

---

### ⚙️ Paso 3 — Preprocesamiento

**Sección:** `3. Preprocesamiento`

1. **Selecciona la variable objetivo** (target) para el modelo supervisado.
2. **Excluye columnas** irrelevantes (IDs, fechas, texto libre).
3. Configura el **pipeline automático**:
   - Estrategia de imputación de nulos (mediana, media, moda, constante)
   - Tipo de escalado (StandardScaler, MinMaxScaler, RobustScaler)
   - Codificación categórica (OneHotEncoding, Label Encoding, Ordinal)
   - Reducción dimensional con PCA (opcional)
4. Haz clic en **"⚙️ Aplicar Preprocesamiento"**.
5. Revisa la pestaña **"📊 Comparación"** para ver antes vs. después.

> 🔒 **Anti Data Leakage**: El pipeline se ajusta (`fit`) **exclusivamente** con los datos de entrenamiento y solo se aplica (`transform`) en validación y test.

---

### ✂️ Paso 4 — Partición & Baseline

**Sección:** `4. Partición & Baseline`

1. Define los porcentajes de **Train / Validación / Test** (por defecto 70/15/15).
2. Configura el **random seed** para reproducibilidad.
3. Activa la **estratificación** (recomendado para clases desbalanceadas).
4. El sistema calcula automáticamente un **Baseline** (clasificador por mayoría).
5. Las particiones se guardan en `session_state` para usarse en clasificación.

---

### 🔮 Paso 5 — Clustering

**Sección:** `5. Clustering`

#### Sub-pestaña: ⚙️ Configuración
1. Selecciona las **variables numéricas** a usar para clustering.
2. Define el **K máximo** a explorar y el **tamaño de muestra**.
3. Elige el método de visualización 2D (**PCA** o **t-SNE**).
4. Haz clic en **"✅ Preparar Datos para Clustering"**.

#### Sub-pestaña: 📊 Selección de K
1. Haz clic en **"📊 Calcular Método del Codo + Silhouette"**.
2. Analiza el gráfico para elegir el K óptimo.
3. El sistema sugiere automáticamente el mejor K.

#### Sub-pestaña: 🗂️ K-Means
1. Ajusta el número de clusters K (usa el sugerido como punto de partida).
2. Configura parámetros avanzados (inicialización, iteraciones).
3. Haz clic en **"🚀 Entrenar K-Means"**.

#### Sub-pestaña: 🌳 Jerárquico
1. Selecciona el número de clusters y el método de enlace.
2. Haz clic en **"🌳 Entrenar Clustering Jerárquico"**.
3. Visualiza el **dendrograma** generado.

#### Sub-pestañas: 🗺️ Visualización y 📋 Perfiles
- Visualiza los clusters en 2D con PCA o t-SNE.
- Analiza los **perfiles de cada cluster** con heatmap y medias por variable.
- Lee la **interpretación automática** generada por el sistema.

---

### 🌳 Paso 6 — Clasificación

**Sección:** `6. Clasificación`

> ⚠️ Requiere haber completado la **Partición de Datos** (Paso 4).

#### Sub-pestaña: ⚙️ Configuración
- Define el número de **folds** para validación cruzada (K-Fold, 3-10).
- Selecciona la **métrica principal** de optimización.

#### Sub-pestaña: 🌲 Árbol de Decisión
1. Configura hiperparámetros: profundidad máxima, muestras mínimas, criterio.
2. Haz clic en **"🌲 Entrenar Árbol de Decisión"**.
3. Analiza: métricas, importancia de variables, visualización del árbol.

#### Sub-pestaña: 🌳 Random Forest
1. Configura: número de árboles, profundidad, features por árbol.
2. Haz clic en **"🌳 Entrenar Random Forest"**.
3. Analiza: métricas, importancia de variables, OOB Score.

#### Sub-pestaña: 📊 Resultados
- Tabla comparativa de todos los modelos entrenados.
- Gráfico de barras por métrica.
- Validación cruzada con error estándar.

---

### 📉 Paso 7 — Evaluación

**Sección:** `7. Evaluación`

| Sub-pestaña | Contenido |
|---|---|
| 🟥 Matriz de Confusión | Heatmap interactivo + reporte de clasificación completo |
| 📈 Curvas ROC | Multi-modelo en el mismo gráfico + AUC por modelo |
| 🏆 Comparación | Radar chart multi-métrica + ranking de modelos |
| 🧪 Evaluación Final | Evalúa el mejor modelo **una sola vez** en el conjunto de Test |

> ⚠️ **Evaluación en Test**: Solo usar cuando estés seguro del modelo final. Evaluar múltiples veces en test constituye data leakage.

---

### 💬 Paso 8 — Interpretación & Exportación

**Sección:** `8. Interpretación`
- Resumen ejecutivo auto-generado del análisis completo.
- Recomendaciones personalizadas basadas en los resultados.
- Análisis del mejor modelo con métricas.

**Sección:** `9. Exportar`
- 📄 **PDF**: Reporte ejecutivo completo con gráficos.
- 📊 **Excel**: Multi-hoja con datos originales, procesados, métricas y predicciones.
- 📁 **CSV**: Predicciones del modelo para nuevos datos.
- 🔧 **JSON**: Configuración completa del experimento.

---

## 📦 Módulos del Sistema

### `app.py` — Punto de Entrada
- Configura la página de Streamlit.
- Renderiza el **sidebar** con navegación y progreso.
- Implementa el **router** que dirige a cada sección.
- Maneja el **session state** global.

### `modules/data_loader.py`
- `load_file(file)`: Carga CSV (con detección automática de delimitador y encoding) o Excel.
- `analyze_dataset(df)`: Detecta tipos de columnas, calcula estadísticas, sugiere targets.
- `convert_numeric_strings(df)`: Convierte columnas de texto que contienen números.

### `modules/eda.py`
- `render_eda(df, meta)`: Módulo completo de exploración con 7 pestañas.
- Genera visualizaciones interactivas con Plotly para distribuciones, correlaciones, outliers y análisis del target.

### `modules/preprocessing.py`
- `render_preprocessing(df, meta)`: Interfaz de configuración del pipeline.
- `apply_preprocessing(df, meta)`: Construye y aplica el `ColumnTransformer + Pipeline` de Scikit-learn.
- `get_preprocessing_config()`: Retorna la configuración actual del pipeline.

### `modules/partitioning.py`
- Divide el dataset en Train/Validación/Test con estratificación opcional.
- Calcula el **Baseline** (ZeroR / DummyClassifier por mayoría).
- Aplica el pipeline de preprocesamiento ajustado solo en train.

### `modules/clustering.py`
- `_tab_config_cluster`: Prepara y normaliza datos con `StandardScaler`.
- `_tab_elbow`: Calcula inertias y Silhouette Scores para K de 2 a K_max.
- `_tab_kmeans`: Entrena `KMeans` con configuración personalizable.
- `_tab_hierarchical`: Entrena `AgglomerativeClustering` + dendrograma.
- `_tab_viz_cluster`: Reducción 2D con PCA o t-SNE + scatter plot.
- `_tab_profiles`: Heatmap de perfiles por cluster + interpretación automática.

### `modules/classification.py`
- `_tab_config_clf`: Configuración de validación cruzada.
- `_tab_decision_tree`: Entrena `DecisionTreeClassifier` + visualización del árbol.
- `_tab_random_forest`: Entrena `RandomForestClassifier` + OOB Score.
- `_evaluate_model(model, ...)`: Calcula Accuracy, Precision, Recall, F1, AUC.
- `_save_model_results(name, metrics)`: Actualiza la tabla comparativa global.

### `modules/evaluation.py`
- `_tab_confusion`: Matriz de confusión + reporte de clasificación.
- `_tab_roc`: Curvas ROC multi-modelo (binario y multiclase OvR).
- `_tab_comparison`: Radar chart + ranking de modelos.
- `_tab_test_eval`: Evaluación final única en el conjunto de test.

### `modules/interpretation.py`
- `generate_executive_summary(meta, metrics, cluster_info)`: Texto ejecutivo del experimento.
- `generate_recommendations(meta, metrics, cluster_info)`: Lista de recomendaciones.
- `interpret_clusters(profile_df, k, cols, method)`: Descripción automática de clusters.
- `interpret_classification_metrics(metrics, model, baseline)`: Análisis comparativo de métricas.

### `modules/export.py`
- `render_export(df, meta)`: Interfaz de exportación con múltiples formatos.
- Genera PDF con FPDF2, Excel con XlsxWriter, JSON de configuración.

### `modules/visualization.py`
- Funciones reutilizables de gráficos con tema oscuro consistente:
  - `cluster_scatter`: Scatter 2D para clusters.
  - `elbow_chart`: Gráfico del codo con Silhouette.
  - `feature_importance_chart`: Barras horizontales de importancia.
  - `confusion_matrix_plot`: Heatmap de confusión.
  - `roc_curve_plot`: Curvas ROC multi-modelo.
  - `model_comparison_bar`: Barras comparativas de modelos.

---

## ⚙️ Configuración de Streamlit

El archivo `.streamlit/config.toml` configura:

```toml
[theme]
base = "dark"
primaryColor = "#7C3AED"        # Morado principal
backgroundColor = "#0F0F1A"     # Fondo oscuro profundo
secondaryBackgroundColor = "#1A1A2E"
textColor = "#E2E8F0"

[server]
maxUploadSize = 200             # Máximo 200 MB por archivo

[browser]
gatherUsageStats = false        # Sin telemetría
```

---

## 🐛 Errores Conocidos y Soluciones

### Error 1: `StreamlitAPIException: st.session_state.<key> cannot be modified`
**Causa**: En Streamlit ≥ 1.30, no se puede escribir manualmente a una clave del `session_state` que ya está vinculada a un widget (mediante `key=`).

**Solución**: No reasignar manualmente el valor después del widget. El widget ya sincroniza su valor automáticamente. Para guardar resultados de entrenamiento, usar claves distintas (ej: `kmeans_k_trained` en lugar de `kmeans_k`).

### Error 2: `NameError: name 'accuracy_score' is not defined`
**Causa**: Las funciones de métricas no estaban importadas a nivel de módulo en `evaluation.py`.

**Solución**: Agregar `accuracy_score, precision_score, recall_score, f1_score, roc_auc_score` al bloque de imports de `sklearn.metrics`.

### Error 3: Advertencia de Kaleido/Plotly incompatibles
**Causa**: Versión de Kaleido 1.x no compatible con Plotly 5.x.

**Solución**: Degradar Kaleido: `pip install kaleido==0.2.1`, o actualizar Plotly: `pip install plotly>=6.1.1`.

### Error 4: `FutureWarning: Styler.applymap has been deprecated`
**Causa**: Pandas >= 2.x deprecó `applymap` en favor de `map`.

**Solución**: Reemplazar `.applymap(...)` por `.map(...)` en `eda.py`.

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## 👨‍💻 Autor

**Eberth Rosales**  
📧 GitHub: [@eberthrosales](https://github.com/eberthrosales)

---

*Desarrollado con ❤️ usando Python + Streamlit + Scikit-learn*
