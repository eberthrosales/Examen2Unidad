"""
data_loader.py
==============
Carga y análisis inicial de datasets CSV o Excel.
Detección automática de tipos de variables.
"""
import io
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Any


# ─── Constantes ────────────────────────────────────────────────────────────────
MAX_CATEGORICAL_UNIQUE = 50   # más de esto → tratar como texto libre
MAX_CATEGORICAL_RATIO  = 0.10 # si únicos/total < 10% → considerar categórica
DATE_SAMPLE_N          = 200  # filas para detectar fechas (perf)


# ─── Funciones de Carga ────────────────────────────────────────────────────────
def load_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Carga un archivo CSV o Excel y retorna (DataFrame, mensaje_error).
    Intenta múltiples delimitadores para CSV.
    """
    filename = uploaded_file.name.lower()
    
    try:
        if filename.endswith(('.xlsx', '.xls')):
            df = _load_excel(uploaded_file)
        elif filename.endswith('.csv'):
            df = _load_csv(uploaded_file)
        else:
            return pd.DataFrame(), "❌ Formato no soportado. Use CSV o Excel (.xlsx, .xls)."
        
        if df is None or df.empty:
            return pd.DataFrame(), "❌ El archivo está vacío o no contiene datos válidos."
        
        # Limpiar nombres de columnas
        df.columns = [str(c).strip() for c in df.columns]
        
        # Eliminar filas completamente vacías
        df = df.dropna(how='all').reset_index(drop=True)
        
        return df, ""
    
    except Exception as e:
        return pd.DataFrame(), f"❌ Error al cargar el archivo: {str(e)}"


def _load_excel(uploaded_file) -> pd.DataFrame:
    """Carga Excel con manejo de múltiples hojas."""
    xlsx = pd.ExcelFile(uploaded_file)
    sheet_names = xlsx.sheet_names
    
    if len(sheet_names) > 1:
        # Guardar en session state para selección posterior
        st.session_state['excel_sheets'] = sheet_names
        sheet = st.selectbox(
            "📋 Selecciona la hoja de trabajo:",
            options=sheet_names,
            key="excel_sheet_select"
        )
    else:
        sheet = sheet_names[0]
    
    return pd.read_excel(uploaded_file, sheet_name=sheet)


def _load_csv(uploaded_file) -> pd.DataFrame:
    """Detecta automáticamente el delimitador del CSV."""
    content = uploaded_file.read()
    
    # Intentar diferentes delimitadores
    delimiters = [',', ';', '\t', '|', ':']
    best_df = None
    best_cols = 0
    
    for delim in delimiters:
        try:
            df = pd.read_csv(
                io.BytesIO(content),
                delimiter=delim,
                encoding='utf-8',
                low_memory=False
            )
            if len(df.columns) > best_cols:
                best_cols = len(df.columns)
                best_df = df
        except Exception:
            try:
                df = pd.read_csv(
                    io.BytesIO(content),
                    delimiter=delim,
                    encoding='latin-1',
                    low_memory=False
                )
                if len(df.columns) > best_cols:
                    best_cols = len(df.columns)
                    best_df = df
            except Exception:
                continue
    
    return best_df


# ─── Análisis de Metadatos ────────────────────────────────────────────────────
def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analiza el DataFrame y retorna un diccionario de metadatos.
    Detecta automáticamente tipos de variables.
    """
    n_rows, n_cols = df.shape
    
    # Categorizar columnas
    numeric_cols      = []
    categorical_cols  = []
    datetime_cols     = []
    text_cols         = []
    id_like_cols      = []
    
    for col in df.columns:
        col_data = df[col].dropna()
        n_unique  = col_data.nunique()
        n_total   = len(col_data)
        
        if n_total == 0:
            continue
        
        dtype = df[col].dtype
        
        # Intentar parsear como fecha
        if _is_datetime_column(col_data, col):
            datetime_cols.append(col)
            continue
        
        # Numérico nativo
        if pd.api.types.is_numeric_dtype(dtype):
            # ¿Es ID-like? (unique ratio alto y es entero)
            if n_unique == n_total and pd.api.types.is_integer_dtype(dtype) and n_total > 100:
                id_like_cols.append(col)
            else:
                numeric_cols.append(col)
            continue
        
        # Object / string
        unique_ratio = n_unique / n_total if n_total > 0 else 1.0
        
        if n_unique <= MAX_CATEGORICAL_UNIQUE or unique_ratio <= MAX_CATEGORICAL_RATIO:
            categorical_cols.append(col)
        else:
            # ¿Podría ser numérico disfrazado?
            try:
                pd.to_numeric(col_data, errors='raise')
                numeric_cols.append(col)
            except Exception:
                text_cols.append(col)
    
    # Estadísticas de nulos
    null_counts   = df.isnull().sum().to_dict()
    null_pct      = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    
    # Columnas objetivo potenciales (categóricas con clases razonables)
    potential_targets = [
        c for c in categorical_cols
        if 2 <= df[c].nunique() <= 20
    ] + [
        c for c in numeric_cols
        if df[c].nunique() <= 20
    ]
    
    # Columnas útiles para ML (excluir IDs y texto libre)
    ml_usable = [c for c in (numeric_cols + categorical_cols)
                 if c not in id_like_cols]
    
    return {
        'n_rows':            n_rows,
        'n_cols':            n_cols,
        'numeric_cols':      numeric_cols,
        'categorical_cols':  categorical_cols,
        'datetime_cols':     datetime_cols,
        'text_cols':         text_cols,
        'id_like_cols':      id_like_cols,
        'null_counts':       null_counts,
        'null_pct':          null_pct,
        'has_nulls':         any(v > 0 for v in null_counts.values()),
        'potential_targets': potential_targets,
        'ml_usable_cols':    ml_usable,
        'dtypes':            df.dtypes.astype(str).to_dict(),
        'memory_mb':         round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }


def _is_datetime_column(col_data: pd.Series, col_name: str) -> bool:
    """Detecta si una columna es de tipo fecha/hora."""
    # Palabras clave en el nombre
    date_keywords = ['date', 'fecha', 'time', 'tiempo', 'year', 'año', 'mes',
                     'month', 'day', 'dia', 'timestamp', 'created', 'updated']
    name_hint = any(k in col_name.lower() for k in date_keywords)
    
    sample = col_data.head(DATE_SAMPLE_N).astype(str)
    try:
        parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
        parse_ratio = parsed.notna().sum() / len(sample)
        return parse_ratio > 0.7 or (name_hint and parse_ratio > 0.3)
    except Exception:
        return False


def get_column_summary(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Resumen estadístico de una columna individual."""
    col_data = df[col].dropna()
    dtype = df[col].dtype
    
    summary = {
        'name':        col,
        'dtype':       str(dtype),
        'n_total':     len(df),
        'n_valid':     len(col_data),
        'n_null':      df[col].isnull().sum(),
        'null_pct':    round(df[col].isnull().sum() / len(df) * 100, 2),
        'n_unique':    col_data.nunique(),
        'unique_pct':  round(col_data.nunique() / len(col_data) * 100, 2) if len(col_data) > 0 else 0,
    }
    
    if pd.api.types.is_numeric_dtype(dtype):
        summary.update({
            'mean':   round(float(col_data.mean()), 4),
            'std':    round(float(col_data.std()), 4),
            'min':    round(float(col_data.min()), 4),
            'q25':    round(float(col_data.quantile(0.25)), 4),
            'median': round(float(col_data.median()), 4),
            'q75':    round(float(col_data.quantile(0.75)), 4),
            'max':    round(float(col_data.max()), 4),
            'skew':   round(float(col_data.skew()), 4),
            'kurt':   round(float(col_data.kurtosis()), 4),
        })
    else:
        top_values = col_data.value_counts().head(5)
        summary.update({
            'top_values': top_values.to_dict(),
            'mode':       col_data.mode()[0] if len(col_data.mode()) > 0 else None,
        })
    
    return summary


def convert_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Intenta convertir columnas objeto que contienen números."""
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            # Si más del 80% se convirtió, aplicar conversión
            if converted.notna().sum() / len(df) > 0.8:
                df[col] = converted
        except Exception:
            pass
    return df
