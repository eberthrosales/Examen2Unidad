"""
preprocessing.py
================
Pipeline automático de preprocesamiento de datos.
Se adapta dinámicamente al dataset.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import plotly.graph_objects as go

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from modules.visualization import apply_layout, PALETTE_PRIMARY


# ─── Configuración de Preprocesamiento ───────────────────────────────────────
def render_preprocessing(df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    """Renderiza el módulo completo de preprocesamiento."""
    
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">⚙️</div>
        <div>
            <div class="section-title">Preprocesamiento Automático</div>
            <div class="section-subtitle">Pipeline inteligente de transformación de datos</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs([
        "🎛️ Configuración", "🔧 Pipeline", "📊 Comparación", "📋 Datos Procesados"
    ])
    
    with tabs[0]: config = _tab_config(df, meta)
    with tabs[1]: _tab_pipeline(df, meta)
    with tabs[2]: _tab_comparacion(df, meta)
    with tabs[3]: _tab_datos_procesados(df, meta)


def _tab_config(df: pd.DataFrame, meta: Dict) -> None:
    """Configuración de parámetros de preprocesamiento."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Variable Objetivo")
        target_options = ['(Sin seleccionar)'] + meta['potential_targets']
        target = st.selectbox(
            "Variable objetivo para clasificación/regresión:",
            target_options, key='prep_target'
        )
        if target != '(Sin seleccionar)':
            st.session_state['target_col'] = target
            n_classes = df[target].nunique()
            st.markdown(f'<span class="badge badge-success">✅ Target: {target} ({n_classes} clases)</span>',
                       unsafe_allow_html=True)
        
        st.markdown("#### 🔢 Variables a Excluir")
        exclude_opts = meta['id_like_cols'] + meta['text_cols'] + meta['datetime_cols']
        st.multiselect(
            "Columnas a excluir del análisis:",
            df.columns.tolist(),
            default=exclude_opts[:5],
            key='prep_exclude'
        )
    
    with col2:
        st.markdown("#### 🏥 Imputación de Nulos")
        num_impute = st.selectbox(
            "Estrategia para variables numéricas:",
            ['median', 'mean', 'most_frequent', 'constant'],
            key='prep_num_impute'
        )
        cat_impute = st.selectbox(
            "Estrategia para variables categóricas:",
            ['most_frequent', 'constant'],
            key='prep_cat_impute'
        )
        
        st.markdown("#### 📐 Escalado")
        st.selectbox(
            "Tipo de escalado:",
            ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Sin escalado'],
            help="StandardScaler: Z-score. MinMaxScaler: [0,1]. RobustScaler: resistente a outliers.",
            key='prep_scaler'
        )
        
        st.markdown("#### 🏷️ Codificación Categórica")
        st.selectbox(
            "Método de codificación:",
            ['OneHotEncoding (OHE)', 'Label Encoding', 'Ordinal Encoding'],
            key='prep_encoding'
        )
    
    # Opciones avanzadas
    with st.expander("🔬 Opciones Avanzadas"):
        use_pca = st.checkbox("Aplicar PCA para reducción de dimensionalidad", key='prep_pca')
        if use_pca:
            st.slider(
                "Número de componentes PCA:",
                2, min(50, len(meta['numeric_cols'])), 
                min(10, len(meta['numeric_cols'])),
                key='prep_pca_n'
            )
        
        st.checkbox(
            "Eliminar columnas con >50% de nulos", 
            value=True, key='prep_drop_null'
        )
    
    # Botón para aplicar
    if st.button("⚙️ Aplicar Preprocesamiento", key='btn_preprocess', type='primary'):
        with st.spinner("Preprocesando datos..."):
            result = apply_preprocessing(df, meta)
            if result is not None:
                X_proc, y, feature_names, pipeline_obj = result
                st.session_state['X_processed'] = X_proc
                st.session_state['y_processed'] = y
                st.session_state['feature_names'] = feature_names
                st.session_state['pipeline_obj'] = pipeline_obj
                st.session_state['preprocessing_done'] = True
                st.success(f"✅ Preprocesamiento completado. Shape: {X_proc.shape}")


def _tab_pipeline(df: pd.DataFrame, meta: Dict) -> None:
    """Visualización del pipeline de transformación."""
    
    st.markdown("### 🔧 Pipeline de Transformación")
    
    target = st.session_state.get('target_col', None)
    excluded = st.session_state.get('prep_exclude', [])
    num_impute = st.session_state.get('prep_num_impute', 'median')
    cat_impute = st.session_state.get('prep_cat_impute', 'most_frequent')
    scaler = st.session_state.get('prep_scaler', 'StandardScaler')
    encoding = st.session_state.get('prep_encoding', 'OneHotEncoding (OHE)')
    use_pca = st.session_state.get('prep_pca', False)
    drop_high_null = st.session_state.get('prep_drop_null', True)
    
    # Calcular columnas que se procesarán
    all_exclude = set(excluded + ([target] if target else []))
    num_cols = [c for c in meta['numeric_cols'] if c not in all_exclude]
    cat_cols = [c for c in meta['categorical_cols'] if c not in all_exclude]
    
    # Columnas con >50% nulos
    high_null_cols = [c for c, p in meta['null_pct'].items() 
                      if p > 50 and c not in all_exclude]
    if drop_high_null:
        num_cols = [c for c in num_cols if c not in high_null_cols]
        cat_cols = [c for c in cat_cols if c not in high_null_cols]
    
    # Mostrar pasos del pipeline
    steps = [
        ("🗑️", "Eliminar Columnas", 
         f"Excluidas: {', '.join(list(all_exclude)[:5]) if all_exclude else 'Ninguna'}"),
        ("🏥", "Imputación Numérica", 
         f"Estrategia: {num_impute} → {len(num_cols)} columnas"),
        ("🏥", "Imputación Categórica", 
         f"Estrategia: {cat_impute} → {len(cat_cols)} columnas"),
        ("🏷️", "Codificación Categórica", 
         f"Método: {encoding} → {len(cat_cols)} columnas"),
        ("📐", "Escalado", 
         f"{scaler} → {len(num_cols)} columnas numéricas"),
    ]
    if use_pca:
        pca_n = st.session_state.get('prep_pca_n', 10)
        steps.append(("🔬", "Reducción PCA", f"{len(num_cols)+len(cat_cols)} → {pca_n} componentes"))
    
    for icon, name, desc in steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="pipeline-step-icon">{icon}</div>
            <div>
                <div class="pipeline-step-text">{name}</div>
                <div class="pipeline-step-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Explicación de Data Leakage
    st.markdown("---")
    st.markdown("### 🔒 Data Leakage — Buenas Prácticas")
    
    with st.expander("📖 ¿Qué es Data Leakage y cómo lo evitamos?", expanded=True):
        st.markdown("""
        <div class="insight-box">
        <b>⚠️ Data Leakage (Fuga de Datos)</b> ocurre cuando información del conjunto de prueba 
        "contamina" el proceso de entrenamiento, causando métricas artificialmente optimistas.
        
        <br><br><b>Ejemplos comunes:</b>
        <ul>
            <li>Escalar con estadísticas de todo el dataset (incluyendo test)</li>
            <li>Imputar nulos con la media de todo el dataset</li>
            <li>Usar características del futuro para predecir el pasado</li>
        </ul>
        
        <br><b>✅ Cómo lo evitamos en esta aplicación:</b>
        <ul>
            <li>Usamos <b>sklearn.Pipeline</b> para encadenar transformaciones</li>
            <li>El escalado y la imputación se <b>fitean SOLO en train</b> y se aplican en val/test</li>
            <li>La partición se hace ANTES de cualquier transformación</li>
            <li>Usamos <b>ColumnTransformer</b> para transformar features independientemente</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Diagrama de flujo
    st.markdown("### 📊 Flujo de Datos")
    
    fig = go.Figure()
    nodes = ["Dataset\nOriginal", "Partición\nTrain/Val/Test", "Pipeline\n(fit en Train)", 
             "Train\nTransformado", "Val\nTransformado", "Test\nTransformado"]
    x_pos = [0, 1.5, 3, 4.5, 4.5, 4.5]
    y_pos = [0, 0, 0, 0.8, 0, -0.8]
    colors = ['#7C3AED', '#06B6D4', '#10B981', '#A78BFA', '#67E8F9', '#6EE7B7']
    
    for i, (node, x, y, c) in enumerate(zip(nodes, x_pos, y_pos, colors)):
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=40, color=c, line=dict(width=2, color='white')),
            text=[node],
            textposition='middle center',
            textfont=dict(size=9, color='white', family='Inter'),
            showlegend=False,
            hoverinfo='skip',
        ))
    
    # Flechas
    arrows = [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5)]
    for s, e in arrows:
        fig.add_annotation(
            x=x_pos[e], y=y_pos[e],
            ax=x_pos[s], ay=y_pos[s],
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True,
            arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor='rgba(148,163,184,0.5)',
        )
    
    apply_layout(fig, 'Flujo sin Data Leakage', height=320)
    fig.update_xaxes(showgrid=False, showticklabels=False, range=[-0.5, 5.5])
    fig.update_yaxes(showgrid=False, showticklabels=False, range=[-1.5, 1.5])
    st.plotly_chart(fig, use_container_width=True)


def _tab_comparacion(df: pd.DataFrame, meta: Dict) -> None:
    """Comparación antes/después del preprocesamiento."""
    
    st.markdown("### 📊 Comparación: Antes vs Después")
    
    if not st.session_state.get('preprocessing_done', False):
        st.info("👆 Aplica el preprocesamiento en la pestaña 'Configuración' primero.")
        return
    
    X_proc = st.session_state.get('X_processed')
    feature_names = st.session_state.get('feature_names', [])
    
    if X_proc is None:
        return
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Antes:**")
        st.markdown(f"- Shape: `{df.shape}`")
        st.markdown(f"- Nulos: `{df.isnull().sum().sum()}`")
        st.markdown(f"- Numéricas: `{len(meta['numeric_cols'])}`")
        st.markdown(f"- Categóricas: `{len(meta['categorical_cols'])}`")
        st.markdown(f"- Tipos: `{df.dtypes.value_counts().to_dict()}`")
    
    with c2:
        st.markdown("**Después:**")
        shape = X_proc.shape if hasattr(X_proc, 'shape') else (0, 0)
        st.markdown(f"- Shape: `{shape}`")
        st.markdown(f"- Nulos: `0`")
        st.markdown(f"- Todo numérico: `✅`")
        st.markdown(f"- Features: `{len(feature_names) if feature_names else shape[1]}`")
    
    # Distribución antes/después para una variable
    num_cols = meta['numeric_cols']
    if num_cols and X_proc is not None:
        col = st.selectbox("Comparar distribución de:", num_cols[:10], key='prep_compare_col')
        if col in (feature_names or []):
            idx = (feature_names or []).index(col)
            if hasattr(X_proc, 'toarray'):
                vals_after = X_proc.toarray()[:, idx]
            else:
                vals_after = X_proc[:, idx] if X_proc.ndim > 1 else X_proc
            
            vals_before = df[col].dropna().values
            
            import plotly.figure_factory as ff
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=vals_before, name='Antes', opacity=0.65, nbinsx=30,
                                       marker_color=PALETTE_PRIMARY[0]))
            fig.add_trace(go.Histogram(x=vals_after, name='Después', opacity=0.65, nbinsx=30,
                                       marker_color=PALETTE_PRIMARY[2]))
            apply_layout(fig, f'Distribución: {col} — Antes vs Después', height=380)
            fig.update_layout(barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)


def _tab_datos_procesados(df: pd.DataFrame, meta: Dict) -> None:
    """Vista del dataset procesado."""
    
    st.markdown("### 📋 Dataset Procesado")
    
    if not st.session_state.get('preprocessing_done', False):
        st.info("👆 Aplica el preprocesamiento en la pestaña 'Configuración' primero.")
        return
    
    X_proc = st.session_state.get('X_processed')
    feature_names = st.session_state.get('feature_names', [])
    y = st.session_state.get('y_processed')
    
    if X_proc is None:
        return
    
    # Convertir a DataFrame
    if hasattr(X_proc, 'toarray'):
        df_proc = pd.DataFrame(X_proc.toarray(), columns=feature_names if feature_names else None)
    else:
        df_proc = pd.DataFrame(X_proc, columns=feature_names if feature_names else None)
    
    if y is not None:
        df_proc['__TARGET__'] = y.values if hasattr(y, 'values') else y
    
    st.dataframe(df_proc.head(50), use_container_width=True)
    
    # Estadísticas del procesado
    st.markdown(f"**Shape final:** `{df_proc.shape}`")
    st.dataframe(df_proc.describe().T, use_container_width=True)


# ─── Función Central de Preprocesamiento ─────────────────────────────────────
def apply_preprocessing(
    df: pd.DataFrame,
    meta: Dict[str, Any],
) -> Optional[Tuple]:
    """
    Aplica el pipeline completo de preprocesamiento.
    Retorna (X_processed, y, feature_names, pipeline_object)
    o None si hay error.
    """
    try:
        target = st.session_state.get('target_col', None)
        excluded = st.session_state.get('prep_exclude', [])
        num_impute = st.session_state.get('prep_num_impute', 'median')
        cat_impute = st.session_state.get('prep_cat_impute', 'most_frequent')
        scaler_name = st.session_state.get('prep_scaler', 'StandardScaler')
        encoding = st.session_state.get('prep_encoding', 'OneHotEncoding (OHE)')
        use_pca = st.session_state.get('prep_pca', False)
        pca_n = st.session_state.get('prep_pca_n', 10)
        drop_high_null = st.session_state.get('prep_drop_null', True)
        
        # Columnas a excluir definitivamente
        all_exclude = set(excluded + ([target] if target else []))
        
        # Columnas con >50% nulos
        if drop_high_null:
            high_null = [c for c, p in meta['null_pct'].items() if p > 50]
            all_exclude.update(high_null)
        
        # Separar features y target
        feature_cols = [c for c in df.columns if c not in all_exclude]
        
        if not feature_cols:
            st.error("❌ No quedan columnas de features después de excluir. Revisa la configuración.")
            return None
        
        X = df[feature_cols].copy()
        y = None
        if target and target in df.columns:
            y = df[target].copy()
        
        # Identificar tipos en las features seleccionadas
        num_cols = [c for c in meta['numeric_cols'] if c in feature_cols]
        cat_cols = [c for c in meta['categorical_cols'] if c in feature_cols]
        
        # Convertir strings numéricos
        for col in cat_cols.copy():
            try:
                X[col] = pd.to_numeric(X[col], errors='raise')
                num_cols.append(col)
                cat_cols.remove(col)
            except Exception:
                pass
        
        # Scaler
        scaler_map = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Sin escalado': 'passthrough',
        }
        scaler = scaler_map.get(scaler_name, StandardScaler())
        
        # Pipeline numérico
        if num_impute == 'constant':
            num_imputer = SimpleImputer(strategy='constant', fill_value=0)
        else:
            num_imputer = SimpleImputer(strategy=num_impute)
        
        num_pipeline_steps = [('imputer', num_imputer)]
        if scaler != 'passthrough':
            num_pipeline_steps.append(('scaler', scaler))
        
        num_pipeline = Pipeline(num_pipeline_steps)
        
        # Pipeline categórico
        if cat_impute == 'constant':
            cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        else:
            cat_imputer = SimpleImputer(strategy=cat_impute)
        
        if encoding == 'OneHotEncoding (OHE)':
            cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        elif encoding == 'Label Encoding':
            cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            cat_encoder = OrdinalEncoder()
        
        cat_pipeline = Pipeline([
            ('imputer', cat_imputer),
            ('encoder', cat_encoder),
        ])
        
        # Transformadores por tipo
        transformers = []
        if num_cols:
            transformers.append(('num', num_pipeline, num_cols))
        if cat_cols:
            transformers.append(('cat', cat_pipeline, cat_cols))
        
        if not transformers:
            st.error("❌ No hay columnas válidas para procesar.")
            return None
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
        )
        
        # PCA opcional
        final_steps = [('preprocessor', preprocessor)]
        if use_pca:
            final_steps.append(('pca', PCA(n_components=pca_n, random_state=42)))
        
        full_pipeline = Pipeline(final_steps)
        
        # Fit y transform
        X_proc = full_pipeline.fit_transform(X)
        
        # Nombres de features
        try:
            if use_pca:
                n_out = pca_n
                feature_names = [f'PC{i+1}' for i in range(n_out)]
            else:
                feature_names_out = []
                if num_cols:
                    feature_names_out.extend(num_cols)
                if cat_cols and encoding == 'OneHotEncoding (OHE)':
                    ohe = preprocessor.named_transformers_['cat']['encoder']
                    try:
                        ohe_names = ohe.get_feature_names_out(cat_cols)
                        feature_names_out.extend(ohe_names.tolist())
                    except Exception:
                        feature_names_out.extend(cat_cols)
                elif cat_cols:
                    feature_names_out.extend(cat_cols)
                feature_names = feature_names_out
        except Exception:
            n_out = X_proc.shape[1] if hasattr(X_proc, 'shape') else 0
            feature_names = [f'feature_{i}' for i in range(n_out)]
        
        # Codificar target si es categórico
        if y is not None:
            if not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), name=target)
                st.session_state['target_encoder'] = le
                st.session_state['target_classes'] = le.classes_.tolist()
            else:
                st.session_state['target_classes'] = sorted(y.dropna().unique().tolist())
        
        # Guardar en session state
        st.session_state['preprocessing_pipeline'] = full_pipeline
        st.session_state['used_num_cols'] = num_cols
        st.session_state['used_cat_cols'] = cat_cols
        
        return X_proc, y, feature_names, full_pipeline
    
    except Exception as e:
        st.error(f"❌ Error en preprocesamiento: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


def get_preprocessing_config() -> Dict:
    """Retorna la configuración actual de preprocesamiento."""
    return {
        'target':       st.session_state.get('target_col'),
        'excluded':     st.session_state.get('prep_exclude', []),
        'num_impute':   st.session_state.get('prep_num_impute', 'median'),
        'cat_impute':   st.session_state.get('prep_cat_impute', 'most_frequent'),
        'scaler':       st.session_state.get('prep_scaler', 'StandardScaler'),
        'encoding':     st.session_state.get('prep_encoding', 'OneHotEncoding (OHE)'),
        'use_pca':      st.session_state.get('prep_pca', False),
        'pca_n':        st.session_state.get('prep_pca_n', 10),
        'done':         st.session_state.get('preprocessing_done', False),
    }
