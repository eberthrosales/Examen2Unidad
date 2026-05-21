"""
app.py
======
ML & Data Science Platform — Aplicación principal de Streamlit.
Plataforma dinámica, modular y adaptable a cualquier dataset tabular.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# ─── Configuración de página (DEBE SER LO PRIMERO) ───────────────────────────
st.set_page_config(
    page_title="ML & Data Science Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "**ML & Data Science Platform** — Plataforma profesional de análisis y modelado."
    }
)

# ─── CSS Personalizado ────────────────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─── Importar Módulos ─────────────────────────────────────────────────────────
from modules.data_loader import load_file, analyze_dataset, convert_numeric_strings
from modules.eda import render_eda
from modules.preprocessing import render_preprocessing
from modules.partitioning import render_partitioning
from modules.clustering import render_clustering
from modules.classification import render_classification
from modules.evaluation import render_evaluation
from modules.interpretation import generate_recommendations, generate_executive_summary
from modules.export import render_export


# ─── Inicialización del Session State ────────────────────────────────────────
def init_session_state():
    defaults = {
        'df':                   None,
        'meta':                 None,
        'file_loaded':          False,
        'preprocessing_done':   False,
        'split_done':           False,
        'cluster_data_ready':   False,
        'kmeans_done':          False,
        'hier_done':            False,
        'dt_done':              False,
        'rf_done':              False,
        'target_col':           None,
        'excluded_cols':        [],
        'trained_models':       {},
        'model_results_df':     None,
        'current_section':      'inicio',
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo / Branding
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
            <div style="font-size:2.5rem; margin-bottom:0.3rem;">🧠</div>
            <div style="font-size:1.1rem; font-weight:800; 
                        background:linear-gradient(135deg,#A78BFA,#67E8F9);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text;">ML Platform</div>
            <div style="font-size:0.7rem; color:#64748B; margin-top:0.2rem;">
                Data Science Suite v1.0
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Status del dataset
        if st.session_state.get('file_loaded'):
            meta = st.session_state.get('meta', {})
            df   = st.session_state.get('df')
            st.markdown(f"""
            <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
                        border-radius:8px; padding:0.6rem 0.8rem; margin-bottom:1rem;">
                <div style="color:#6EE7B7; font-size:0.75rem; font-weight:600;">✅ DATASET CARGADO</div>
                <div style="color:#E2E8F0; font-size:0.8rem; margin-top:0.2rem;">
                    📋 {meta.get('n_rows',0):,} filas × {meta.get('n_cols',0)} columnas
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3);
                        border-radius:8px; padding:0.6rem 0.8rem; margin-bottom:1rem;">
                <div style="color:#FCD34D; font-size:0.75rem; font-weight:600;">⏳ SIN DATASET</div>
                <div style="color:#94A3B8; font-size:0.8rem; margin-top:0.2rem;">
                    Carga un archivo CSV o Excel
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navegación
        st.markdown('<div style="font-size:0.7rem; color:#64748B; font-weight:600; letter-spacing:0.1em; margin-bottom:0.5rem;">NAVEGACIÓN</div>', unsafe_allow_html=True)
        
        sections = [
            ("🏠", "inicio",         "Inicio"),
            ("📂", "carga",          "1. Carga de Datos"),
            ("🔍", "eda",            "2. Exploración (EDA)"),
            ("⚙️", "preprocesamiento", "3. Preprocesamiento"),
            ("✂️", "particion",      "4. Partición & Baseline"),
            ("🔮", "clustering",     "5. Clustering"),
            ("🌳", "clasificacion",  "6. Clasificación"),
            ("📉", "evaluacion",     "7. Evaluación"),
            ("💬", "interpretacion", "8. Interpretación"),
            ("📤", "exportar",       "9. Exportar"),
        ]
        
        current = st.session_state.get('current_section', 'inicio')
        
        for icon, key, label in sections:
            is_current = current == key
            is_locked  = _is_section_locked(key)
            
            if is_locked:
                st.markdown(f"""
                <div style="padding:0.45rem 0.6rem; margin:0.1rem 0; border-radius:8px;
                            color:#4B5563; font-size:0.875rem; cursor:not-allowed;
                            display:flex; align-items:center; gap:0.5rem;">
                    <span>{icon}</span> <span>{label}</span>
                    <span style="margin-left:auto; font-size:0.65rem;">🔒</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                bg = 'linear-gradient(135deg,rgba(124,58,237,0.3),rgba(6,182,212,0.15))' if is_current else 'transparent'
                border = '1px solid rgba(124,58,237,0.4)' if is_current else '1px solid transparent'
                color = '#E2E8F0' if is_current else '#94A3B8'
                
                if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                    st.session_state['current_section'] = key
                    st.rerun()
        
        st.markdown("---")
        
        # Progress tracker
        _render_progress_sidebar()
        
        st.markdown("---")
        
        # Reset
        if st.button("🔄 Reiniciar Todo", key='btn_reset', use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("""
        <div class="app-footer">
            ML Platform v1.0<br>
            Dataset-Agnostic ML Suite
        </div>
        """, unsafe_allow_html=True)


def _is_section_locked(section: str) -> bool:
    """Determina si una sección está bloqueada por falta de prerrequisitos."""
    if section in ('inicio', 'carga'):
        return False
    if not st.session_state.get('file_loaded') and section != 'inicio':
        return True
    if section in ('eda',):
        return not st.session_state.get('file_loaded')
    if section == 'preprocesamiento':
        return not st.session_state.get('file_loaded')
    if section == 'particion':
        return not st.session_state.get('file_loaded')
    if section == 'clustering':
        return not st.session_state.get('file_loaded')
    if section in ('clasificacion', 'evaluacion'):
        return not st.session_state.get('split_done', False)
    if section in ('interpretacion', 'exportar'):
        return not st.session_state.get('file_loaded')
    return False


def _render_progress_sidebar():
    """Muestra el progreso del workflow."""
    steps = [
        ("Datos cargados",     st.session_state.get('file_loaded', False)),
        ("Preprocesamiento",   st.session_state.get('preprocessing_done', False)),
        ("Partición",          st.session_state.get('split_done', False)),
        ("Clustering",         st.session_state.get('kmeans_done') or st.session_state.get('hier_done')),
        ("Clasificación",      st.session_state.get('dt_done') or st.session_state.get('rf_done')),
        ("Evaluación",         bool(st.session_state.get('test_metrics'))),
    ]
    
    completed = sum(1 for _, done in steps if done)
    total = len(steps)
    pct = completed / total
    
    st.markdown(f'<div style="font-size:0.7rem; color:#64748B; font-weight:600; letter-spacing:0.1em; margin-bottom:0.5rem;">PROGRESO ({completed}/{total})</div>', unsafe_allow_html=True)
    st.progress(pct)
    
    for label, done in steps:
        icon = '✅' if done else '⏳'
        color = '#6EE7B7' if done else '#4B5563'
        st.markdown(f'<div style="font-size:0.75rem; color:{color}; padding:0.1rem 0;">{icon} {label}</div>', unsafe_allow_html=True)


# ─── Secciones Principales ────────────────────────────────────────────────────
def render_inicio():
    """Página de inicio / Landing."""
    
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🧠 ML & Data Science Platform</div>
        <div class="hero-subtitle">
            Plataforma profesional de análisis de datos e inteligencia artificial.<br>
            Compatible con <b>cualquier dataset tabular</b> — sin configuración manual.
        </div>
        <br>
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
            <span class="badge badge-primary">🔍 EDA Automático</span>
            <span class="badge badge-info">⚙️ Preprocesamiento Inteligente</span>
            <span class="badge badge-success">🔮 Clustering</span>
            <span class="badge badge-warning">🌳 Clasificación</span>
            <span class="badge badge-danger">📊 Evaluación Comparativa</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features cards
    c1, c2, c3 = st.columns(3)
    
    features = [
        (c1, "🎯", "Detección Automática", 
         "Identifica variables numéricas, categóricas y posibles targets sin configuración."),
        (c2, "⚙️", "Pipeline Inteligente",
         "Sklearn Pipeline anti-leakage: imputación, codificación y escalado automático."),
        (c3, "📊", "Visualizaciones Interactivas",
         "Plotly, Matplotlib y Seaborn con tema oscuro profesional."),
    ]
    
    for col, icon, title, desc in features:
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; padding:1.5rem;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:700; font-size:1rem; color:#A78BFA; margin-bottom:0.5rem;">{title}</div>
                <div style="font-size:0.85rem; color:#94A3B8; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c4, c5, c6 = st.columns(3)
    features2 = [
        (c4, "🔮", "K-Means + Jerárquico",
         "Codo, Silhouette, Dendrogramas, PCA/t-SNE con perfiles automáticos de clusters."),
        (c5, "🌳", "Árbol + Random Forest",
         "Validación cruzada, feature importance, visualización del árbol y curvas ROC."),
        (c6, "📤", "Exportación Completa",
         "PDF, Excel, CSV, JSON. Reporte ejecutivo generado automáticamente."),
    ]
    
    for col, icon, title, desc in features2:
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; padding:1.5rem;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:700; font-size:1rem; color:#06B6D4; margin-bottom:0.5rem;">{title}</div>
                <div style="font-size:0.85rem; color:#94A3B8; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Workflow steps
    st.markdown("### 🗺️ Flujo de Trabajo")
    
    steps = [
        ("📂", "1. Carga", "CSV o Excel"),
        ("🔍", "2. EDA",   "Explora datos"),
        ("⚙️", "3. Preprocesa", "Pipeline auto"),
        ("✂️", "4. Particiona", "Train/Val/Test"),
        ("🔮", "5. Clustering", "Segmenta"),
        ("🌳", "6. Clasifica",  "Modelos ML"),
        ("📉", "7. Evalúa",     "Métricas"),
        ("📤", "8. Exporta",    "Reportes"),
    ]
    
    cols = st.columns(len(steps))
    for i, (col, (icon, label, sub)) in enumerate(zip(cols, steps)):
        with col:
            is_done = _check_step_done(i)
            border_color = '#10B981' if is_done else '#7C3AED'
            st.markdown(f"""
            <div style="text-align:center; padding:0.75rem 0.4rem; 
                        border:1px solid {border_color}; border-radius:10px;
                        background:rgba(26,26,46,0.6);">
                <div style="font-size:1.3rem;">{icon}</div>
                <div style="font-size:0.75rem; font-weight:700; color:#E2E8F0; margin-top:0.3rem;">{label}</div>
                <div style="font-size:0.65rem; color:#64748B;">{sub}</div>
                {'<div style="color:#10B981; font-size:0.7rem;">✅ Listo</div>' if is_done else ''}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # CTA
    if not st.session_state.get('file_loaded'):
        st.markdown("""
        <div style="text-align:center; padding:1.5rem;">
            <div style="font-size:1.2rem; font-weight:600; color:#A78BFA; margin-bottom:0.5rem;">
                ⬆️ Comienza subiendo tu dataset en <b>1. Carga de Datos</b>
            </div>
            <div style="color:#64748B; font-size:0.9rem;">
                Compatible con CSV (separador automático) y Excel (.xlsx, .xls)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Stack tecnológico
    with st.expander("🛠️ Stack Tecnológico"):
        c1, c2, c3, c4 = st.columns(4)
        tech = [
            (c1, ["🐍 Python", "📊 Streamlit", "🐼 Pandas", "🔢 NumPy"]),
            (c2, ["🤖 Scikit-learn", "📈 Plotly", "🎨 Matplotlib", "🌊 Seaborn"]),
            (c3, ["🔬 SciPy", "📋 OpenPyXL", "📄 FPDF2", "💼 XlsxWriter"]),
            (c4, ["🔄 Sklearn Pipeline", "🎯 K-Means / KNN", "🌳 RF / DT", "📉 PCA / t-SNE"]),
        ]
        for col, items in tech:
            with col:
                for item in items:
                    st.markdown(f'<span class="badge badge-primary" style="margin:0.2rem">{item}</span>', 
                               unsafe_allow_html=True)


def _check_step_done(step_idx: int) -> bool:
    checks = [
        st.session_state.get('file_loaded', False),
        st.session_state.get('file_loaded', False),
        st.session_state.get('preprocessing_done', False),
        st.session_state.get('split_done', False),
        st.session_state.get('kmeans_done') or st.session_state.get('hier_done'),
        st.session_state.get('dt_done') or st.session_state.get('rf_done'),
        bool(st.session_state.get('model_results_df') is not None),
        False,
    ]
    return checks[step_idx] if step_idx < len(checks) else False


def render_carga():
    """Módulo de carga de datos."""
    
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📂</div>
        <div>
            <div class="section-title">Carga de Datos</div>
            <div class="section-subtitle">Sube tu dataset CSV o Excel</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload
    uploaded = st.file_uploader(
        "Arrastra o selecciona tu archivo:",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos soportados: CSV (delimitador automático) y Excel (.xlsx, .xls)",
        key='file_uploader'
    )
    
    if uploaded is not None:
        with st.spinner("⏳ Cargando y analizando dataset..."):
            df, error = load_file(uploaded)
            
            if error:
                st.error(error)
                return
            
            # Convertir strings numéricos
            df = convert_numeric_strings(df)
            
            # Analizar
            meta = analyze_dataset(df)
            
            # Guardar en session state
            st.session_state['df']          = df
            st.session_state['meta']        = meta
            st.session_state['file_loaded'] = True
            st.session_state['filename']    = uploaded.name
        
        st.success(f"✅ **{uploaded.name}** cargado exitosamente — {meta['n_rows']:,} filas × {meta['n_cols']} columnas")
        
        # Métricas rápidas
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{meta['n_rows']:,}</div>
                <div class="metric-label">📋 Filas</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{meta['n_cols']}</div>
                <div class="metric-label">📐 Columnas</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{len(meta['numeric_cols'])}</div>
                <div class="metric-label">🔢 Numéricas</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            null_t = sum(meta['null_counts'].values())
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{null_t:,}</div>
                <div class="metric-label">🕳️ Nulos</div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Vista previa
        st.markdown("### 👁️ Vista Previa del Dataset")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Info de columnas
        st.markdown("### 📋 Información de Columnas")
        col_info = pd.DataFrame({
            'Columna': df.columns.tolist(),
            'Tipo Detectado': [
                '🔢 Numérica'   if c in meta['numeric_cols'] else
                '🏷️ Categórica' if c in meta['categorical_cols'] else
                '📅 Fecha'      if c in meta['datetime_cols'] else
                '🆔 ID-Like'    if c in meta['id_like_cols'] else
                '📝 Texto Libre'
                for c in df.columns
            ],
            'Dtype': [str(df[c].dtype) for c in df.columns],
            'Nulos': [meta['null_counts'].get(c, 0) for c in df.columns],
            'Nulos %': [f"{meta['null_pct'].get(c, 0):.1f}%" for c in df.columns],
            'Únicos': [df[c].nunique() for c in df.columns],
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        # Targets sugeridos
        if meta['potential_targets']:
            st.markdown("### 🎯 Variables Objetivo Sugeridas")
            for t in meta['potential_targets']:
                n_u = df[t].nunique()
                st.markdown(f'<span class="badge badge-success">🎯 {t} ({n_u} clases)</span>', 
                           unsafe_allow_html=True)
        
        # Navegar al EDA
        if st.button("➡️ Ir a Análisis Exploratorio (EDA)", type='primary'):
            st.session_state['current_section'] = 'eda'
            st.rerun()
    
    elif st.session_state.get('file_loaded'):
        # Mostrar dataset ya cargado
        df   = st.session_state['df']
        meta = st.session_state['meta']
        fname = st.session_state.get('filename', 'dataset.csv')
        
        st.info(f"📂 Dataset actual: **{fname}** — {meta['n_rows']:,} filas × {meta['n_cols']} columnas")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("🔄 Cargar otro dataset", type='secondary'):
            st.session_state['file_loaded'] = False
            st.rerun()
    
    else:
        # Instrucciones con ejemplos
        st.markdown("""
        <div class="glass-card">
        <h4 style="color:#A78BFA">📖 Instrucciones</h4>
        <ul>
            <li>Sube un archivo <b>CSV</b> o <b>Excel</b> con datos tabulares.</li>
            <li>El sistema detectará automáticamente los tipos de variables.</li>
            <li>Compatible con cualquier estructura de datos.</li>
            <li>No se requieren nombres de columnas específicos.</li>
        </ul>
        <h4 style="color:#06B6D4">✅ Formatos Soportados</h4>
        <ul>
            <li>CSV con cualquier delimitador (,  ;  TAB  |  :) — detección automática</li>
            <li>Excel (.xlsx, .xls) con selección de hoja</li>
            <li>Codificación UTF-8 o Latin-1</li>
        </ul>
        <h4 style="color:#10B981">💡 Consejo</h4>
        <p style="color:#94A3B8">
            Asegúrate de que la primera fila sea el encabezado de columnas. 
            Los datasets deben ser rectangulares (mismas columnas en todas las filas).
        </p>
        </div>
        """, unsafe_allow_html=True)


def render_interpretacion():
    """Módulo de interpretación y conclusiones."""
    
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">💬</div>
        <div>
            <div class="section-title">Interpretación Automática</div>
            <div class="section-subtitle">Conclusiones, recomendaciones y resumen ejecutivo</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    meta = st.session_state.get('meta', {})
    
    tabs = st.tabs(["📋 Resumen Ejecutivo", "💡 Recomendaciones", "🏆 Mejor Modelo"])
    
    with tabs[0]:
        best_model_metrics = st.session_state.get('best_model_metrics')
        cluster_info = st.session_state.get('cluster_info')
        
        summary = generate_executive_summary(meta, best_model_metrics, cluster_info)
        st.markdown(summary)
        
        # Estado del pipeline
        st.markdown("### 📊 Estado del Pipeline")
        pipeline_status = [
            ("📂 Carga de Datos",    st.session_state.get('file_loaded', False)),
            ("⚙️ Preprocesamiento",  st.session_state.get('preprocessing_done', False)),
            ("✂️ Partición",         st.session_state.get('split_done', False)),
            ("🔮 Clustering",        st.session_state.get('kmeans_done') or st.session_state.get('hier_done')),
            ("🌳 Clasificación",     st.session_state.get('dt_done') or st.session_state.get('rf_done')),
            ("🧪 Evaluación Test",   bool(st.session_state.get('test_metrics'))),
        ]
        
        for label, done in pipeline_status:
            emoji = "✅" if done else "⏳"
            color = "#6EE7B7" if done else "#64748B"
            st.markdown(f'<div style="color:{color}; padding:0.3rem 0; font-size:0.9rem;">{emoji} {label}</div>',
                       unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 💡 Recomendaciones Personalizadas")
        
        best_metrics = st.session_state.get('test_metrics') or st.session_state.get('rf_metrics') or st.session_state.get('dt_metrics')
        cluster_info = st.session_state.get('cluster_info')
        
        recs = generate_recommendations(meta, best_metrics, cluster_info)
        
        for rec in recs:
            st.markdown(f'<div class="insight-box">💡 {rec}</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("### 🏆 Análisis del Mejor Modelo")
        
        results_df = st.session_state.get('model_results_df')
        
        if results_df is None or results_df.empty:
            st.info("⚙️ Entrena modelos en la sección de **Clasificación** para ver el análisis.")
            return
        
        from modules.interpretation import interpret_model_comparison
        analysis = interpret_model_comparison(results_df)
        st.markdown(analysis)
        
        # Métricas del mejor modelo
        best_row = results_df.sort_values('F1-Score', ascending=False).iloc[0]
        
        st.markdown("#### 📊 Métricas del Mejor Modelo")
        cols = st.columns(5)
        metrics_show = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        
        for i, m in enumerate(metrics_show):
            val = best_row.get(m, 0)
            color = '#10B981' if val > 0.7 else '#F59E0B' if val > 0.5 else '#EF4444'
            with cols[i]:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:{color}">{val:.3f}</div>
                    <div class="metric-label">{m}</div>
                </div>""", unsafe_allow_html=True)


# ─── Router Principal ─────────────────────────────────────────────────────────
def main():
    render_sidebar()
    
    section = st.session_state.get('current_section', 'inicio')
    df      = st.session_state.get('df')
    meta    = st.session_state.get('meta', {})
    
    # Routing
    if section == 'inicio':
        render_inicio()
    
    elif section == 'carga':
        render_carga()
    
    elif section == 'eda':
        if df is None:
            st.warning("⚠️ Primero carga un dataset en la sección **1. Carga de Datos**.")
        else:
            render_eda(df, meta)
    
    elif section == 'preprocesamiento':
        if df is None:
            st.warning("⚠️ Primero carga un dataset.")
        else:
            render_preprocessing(df, meta)
    
    elif section == 'particion':
        if df is None:
            st.warning("⚠️ Primero carga un dataset.")
        else:
            render_partitioning(df, meta)
    
    elif section == 'clustering':
        if df is None:
            st.warning("⚠️ Primero carga un dataset.")
        else:
            render_clustering(df, meta)
    
    elif section == 'clasificacion':
        if not st.session_state.get('split_done', False):
            st.warning("⚠️ Completa la **Partición de Datos** antes de clasificar.")
        elif df is None:
            st.warning("⚠️ Primero carga un dataset.")
        else:
            render_classification(df, meta)
    
    elif section == 'evaluacion':
        if df is None:
            st.warning("⚠️ Primero carga un dataset.")
        else:
            render_evaluation(df, meta)
    
    elif section == 'interpretacion':
        if df is None:
            st.warning("⚠️ Primero carga un dataset.")
        else:
            render_interpretacion()
    
    elif section == 'exportar':
        if df is None:
            st.warning("⚠️ Primero carga un dataset.")
        else:
            render_export(df, meta)
    
    else:
        render_inicio()


if __name__ == '__main__':
    main()
