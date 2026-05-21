"""
partitioning.py
===============
Partición de datos Train/Validación/Test + Modelo Baseline.
Explicación visual de data leakage y buenas prácticas.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize

from modules.visualization import split_donut, apply_layout, PALETTE_PRIMARY


# ─── Sección Principal ────────────────────────────────────────────────────────
def render_partitioning(df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">✂️</div>
        <div>
            <div class="section-title">Partición de Datos y Modelo Baseline</div>
            <div class="section-subtitle">División estratificada Train / Validación / Test</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["⚙️ Configuración", "📊 Visualización", "🤖 Baseline", "📖 Conceptos"])
    
    with tabs[0]: _tab_config_partition(df, meta)
    with tabs[1]: _tab_visual_partition(df, meta)
    with tabs[2]: _tab_baseline(df, meta)
    with tabs[3]: _tab_conceptos()


# ─── Tab: Configuración ───────────────────────────────────────────────────────
def _tab_config_partition(df: pd.DataFrame, meta: Dict) -> None:
    
    # Verificar preprocesamiento
    if not st.session_state.get('preprocessing_done', False):
        st.warning("⚠️ Primero aplica el **Preprocesamiento** (sección anterior) para continuar.")
        return
    
    X = st.session_state.get('X_processed')
    y = st.session_state.get('y_processed')
    
    if X is None:
        st.error("❌ No se encontraron datos procesados.")
        return
    
    if y is None:
        st.warning("⚠️ No se seleccionó variable objetivo en el preprocesamiento. La partición se realizará solo para clustering.")
    
    n_total = X.shape[0]
    
    st.markdown(f"**Dataset:** `{n_total:,}` muestras × `{X.shape[1]}` features")
    
    # Sliders de partición
    st.markdown("#### ⚖️ Proporciones de Partición")
    
    train_pct = st.slider("Entrenamiento (%)", 50, 85, 70, 5, key='part_train')
    remaining = 100 - train_pct
    val_pct_of_rem = st.slider(f"Validación (del {remaining}% restante, %):", 
                                20, 70, 50, 10, key='part_val')
    
    val_pct  = round(remaining * val_pct_of_rem / 100, 1)
    test_pct = round(remaining * (100 - val_pct_of_rem) / 100, 1)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{train_pct}%</div>
            <div class="metric-label">🟣 Entrenamiento</div>
            <div class="metric-delta positive">{int(n_total * train_pct/100):,} muestras</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{val_pct}%</div>
            <div class="metric-label">🔵 Validación</div>
            <div class="metric-delta positive">{int(n_total * val_pct/100):,} muestras</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{test_pct}%</div>
            <div class="metric-label">🟢 Prueba</div>
            <div class="metric-delta positive">{int(n_total * test_pct/100):,} muestras</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Opciones
    col_a, col_b = st.columns(2)
    with col_a:
        random_seed = st.number_input("Semilla aleatoria:", 0, 9999, 42, key='part_seed')
        stratify = st.checkbox(
            "Estratificación (mantiene proporción de clases)", 
            value=True if y is not None else False,
            disabled=(y is None),
            key='part_stratify'
        )
    with col_b:
        shuffle_data = st.checkbox("Mezclar datos antes de partir", value=True, key='part_shuffle')
        st.markdown(f"**Semilla:** `{random_seed}` → Resultados reproducibles ✅")
    
    # Botón para partir
    if st.button("✂️ Aplicar Partición", key='btn_split', type='primary'):
        with st.spinner("Dividiendo datos..."):
            splits = _apply_split(X, y, train_pct, val_pct, test_pct,
                                  random_seed, stratify and y is not None, shuffle_data)
            if splits:
                X_train, X_val, X_test, y_train, y_val, y_test = splits
                st.session_state.update({
                    'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                    'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                    'split_done': True,
                    'train_pct': train_pct, 'val_pct': val_pct, 'test_pct': test_pct,
                })
                st.success(f"""✅ Partición exitosa:
                - **Train:** {X_train.shape[0]:,} muestras
                - **Validación:** {X_val.shape[0]:,} muestras  
                - **Test:** {X_test.shape[0]:,} muestras""")


def _apply_split(X, y, train_pct, val_pct, test_pct, seed, stratify, shuffle):
    """Aplica la partición train/val/test."""
    try:
        n = X.shape[0]
        test_size = test_pct / 100
        val_size_adj = val_pct / (train_pct + val_pct)
        
        strat_y = y if stratify and y is not None else None
        
        # Primera división: train+val vs test
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y if y is not None else np.zeros(n),
            test_size=test_size,
            random_state=seed,
            shuffle=shuffle,
            stratify=strat_y,
        )
        
        strat_tv = y_tv if stratify and y is not None else None
        
        # Segunda división: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv,
            test_size=val_size_adj,
            random_state=seed,
            shuffle=shuffle,
            stratify=strat_tv,
        )
        
        if y is None:
            y_train = y_val = y_test = None
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    except Exception as e:
        st.error(f"❌ Error en la partición: {str(e)}")
        return None


# ─── Tab: Visualización ───────────────────────────────────────────────────────
def _tab_visual_partition(df: pd.DataFrame, meta: Dict) -> None:
    
    if not st.session_state.get('split_done', False):
        st.info("⚙️ Configura y aplica la partición primero.")
        return
    
    train_pct = st.session_state.get('train_pct', 70)
    val_pct   = st.session_state.get('val_pct', 15)
    test_pct  = st.session_state.get('test_pct', 15)
    n_total   = df.shape[0]
    
    X_train = st.session_state.get('X_train')
    X_val   = st.session_state.get('X_val')
    X_test  = st.session_state.get('X_test')
    y_train = st.session_state.get('y_train')
    
    # Donut chart
    st.plotly_chart(split_donut(train_pct, val_pct, test_pct, n_total), use_container_width=True)
    
    # Distribución de clases por split
    if y_train is not None:
        st.markdown("#### 🎯 Distribución de Clases por Split")
        
        y_val   = st.session_state.get('y_val')
        y_test  = st.session_state.get('y_test')
        classes = st.session_state.get('target_classes', [])
        
        splits_y = {'Train': y_train, 'Validación': y_val, 'Test': y_test}
        fig = go.Figure()
        
        for i, (split_name, y_split) in enumerate(splits_y.items()):
            if y_split is None:
                continue
            y_arr = np.array(y_split)
            unique, counts = np.unique(y_arr, return_counts=True)
            pcts = counts / len(y_arr) * 100
            
            labels = [str(classes[int(u)]) if classes and int(u) < len(classes) 
                     else str(u) for u in unique]
            
            fig.add_trace(go.Bar(
                name=split_name,
                x=labels,
                y=pcts,
                marker_color=PALETTE_PRIMARY[i],
                text=[f'{p:.1f}%' for p in pcts],
                textposition='outside',
            ))
        
        apply_layout(fig, 'Distribución de Clases por Split (Estratificada)', height=400)
        fig.update_layout(barmode='group')
        fig.update_yaxes(title_text='Porcentaje (%)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Verificar estratificación
        if y_val is not None:
            train_dist = pd.Series(y_train).value_counts(normalize=True)
            val_dist   = pd.Series(y_val).value_counts(normalize=True)
            max_diff   = (train_dist - val_dist).abs().max()
            
            if max_diff < 0.05:
                st.markdown('<div class="insight-box">✅ <b>Estratificación exitosa</b>: La distribución de clases es consistente entre splits (diferencia máxima: {:.2f}%).</div>'.format(max_diff*100), unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-box">⚠️ <b>Leve variación</b>: Diferencia de {:.2f}% en la distribución de clases entre splits. Normal en datasets pequeños.</div>'.format(max_diff*100), unsafe_allow_html=True)


# ─── Tab: Baseline ────────────────────────────────────────────────────────────
def _tab_baseline(df: pd.DataFrame, meta: Dict) -> None:
    
    st.markdown("### 🤖 Modelo Baseline")
    
    if not st.session_state.get('split_done', False):
        st.info("⚙️ Aplica la partición primero.")
        return
    
    y_train = st.session_state.get('y_train')
    y_val   = st.session_state.get('y_val')
    X_train = st.session_state.get('X_train')
    X_val   = st.session_state.get('X_val')
    
    if y_train is None:
        st.info("ℹ️ No hay variable objetivo. El baseline requiere una variable objetivo.")
        return
    
    with st.expander("📖 ¿Para qué sirve el Baseline?", expanded=False):
        st.markdown("""
        <div class="insight-box">
        El <b>modelo baseline</b> es el punto de referencia mínimo. Si nuestros modelos avanzados 
        no superan al baseline, algo está mal (data leakage, mal preprocesamiento, target equivocado).
        
        <br><br>Usamos <b>DummyClassifier</b> con estrategia <b>stratified</b>: predice clases 
        al azar respetando las proporciones observadas en el entrenamiento. 
        Es equivalente a un humano sin conocimiento del dominio.
        </div>
        """, unsafe_allow_html=True)
    
    baseline_strategy = st.selectbox(
        "Estrategia del baseline:", 
        ['stratified', 'most_frequent', 'uniform'],
        help="stratified: predice respetando distribución. most_frequent: siempre predice la clase mayoritaria.",
        key='baseline_strategy'
    )
    
    if st.button("🚀 Entrenar Baseline", key='btn_baseline', type='primary'):
        with st.spinner("Entrenando baseline..."):
            metrics = _train_baseline(X_train, y_train, X_val, y_val, baseline_strategy)
            if metrics:
                st.session_state['baseline_metrics'] = metrics
                _display_baseline_metrics(metrics)
    elif st.session_state.get('baseline_metrics'):
        _display_baseline_metrics(st.session_state['baseline_metrics'])


def _train_baseline(X_train, y_train, X_val, y_val, strategy: str) -> Optional[Dict]:
    """Entrena y evalúa el DummyClassifier."""
    try:
        dummy = DummyClassifier(strategy=strategy, random_state=42)
        dummy.fit(X_train, y_train)
        
        y_pred = dummy.predict(X_val)
        
        n_classes = len(np.unique(y_train))
        avg = 'binary' if n_classes == 2 else 'weighted'
        
        metrics = {
            'accuracy':  round(accuracy_score(y_val, y_pred), 4),
            'precision': round(precision_score(y_val, y_pred, average=avg, zero_division=0), 4),
            'recall':    round(recall_score(y_val, y_pred, average=avg, zero_division=0), 4),
            'f1':        round(f1_score(y_val, y_pred, average=avg, zero_division=0), 4),
        }
        
        # AUC
        try:
            if n_classes == 2:
                y_prob = dummy.predict_proba(X_val)[:, 1]
                metrics['auc'] = round(roc_auc_score(y_val, y_prob), 4)
            else:
                y_prob = dummy.predict_proba(X_val)
                classes = sorted(np.unique(y_train))
                y_val_bin = label_binarize(y_val, classes=classes)
                metrics['auc'] = round(roc_auc_score(y_val_bin, y_prob, 
                                                      multi_class='ovr', average='weighted'), 4)
        except Exception:
            metrics['auc'] = 0.5
        
        metrics['model'] = dummy
        metrics['n_classes'] = n_classes
        metrics['strategy'] = strategy
        
        # Guardar modelo entrenado
        st.session_state['trained_models'] = st.session_state.get('trained_models', {})
        st.session_state['trained_models']['Baseline'] = {
            'model': dummy, 'metrics': metrics
        }
        
        return metrics
    
    except Exception as e:
        st.error(f"❌ Error en baseline: {str(e)}")
        return None


def _display_baseline_metrics(metrics: Dict) -> None:
    """Muestra las métricas del baseline visualmente."""
    
    st.markdown("#### 📊 Métricas del Modelo Baseline")
    
    metric_items = [
        ("Accuracy",  metrics.get('accuracy', 0),  "Porcentaje de predicciones correctas"),
        ("Precision", metrics.get('precision', 0), "Exactitud de las predicciones positivas"),
        ("Recall",    metrics.get('recall', 0),    "Sensibilidad para detectar positivos"),
        ("F1-Score",  metrics.get('f1', 0),        "Media armónica de Precision y Recall"),
        ("AUC-ROC",   metrics.get('auc', 0),       "Área bajo la curva ROC"),
    ]
    
    cols = st.columns(5)
    for i, (name, val, tooltip) in enumerate(metric_items):
        with cols[i]:
            color = '#10B981' if val > 0.7 else '#F59E0B' if val > 0.5 else '#EF4444'
            st.markdown(f"""
            <div class="metric-card" title="{tooltip}">
                <div class="metric-value" style="color:{color}">{val:.3f}</div>
                <div class="metric-label">{name}</div>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráfico radar de métricas
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    values = [metrics.get('accuracy', 0), metrics.get('precision', 0),
              metrics.get('recall', 0), metrics.get('f1', 0), metrics.get('auc', 0)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(124,58,237,0.2)',
        line=dict(color=PALETTE_PRIMARY[0], width=2),
        name='Baseline',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], 
                           gridcolor='rgba(148,163,184,0.2)',
                           tickfont=dict(color='#94A3B8')),
            angularaxis=dict(gridcolor='rgba(148,163,184,0.2)'),
            bgcolor='rgba(26,26,46,0.5)',
        ),
        paper_bgcolor='rgba(15,15,26,0)',
        font=dict(color='#E2E8F0'),
        height=380,
        title=dict(text='Radar de Métricas — Baseline', font=dict(color='#A78BFA')),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretación
    acc = metrics.get('accuracy', 0)
    n_classes = metrics.get('n_classes', 2)
    expected_random = 1 / n_classes
    
    st.markdown(f"""
    <div class="insight-box">
    📊 <b>Interpretación del Baseline</b><br><br>
    El modelo baseline obtuvo <b>Accuracy = {acc:.3f}</b>. 
    Un modelo completamente aleatorio esperaría ~{expected_random:.3f} con {n_classes} clases.
    <br><br>
    <b>✅ Objetivo:</b> Tus modelos de Machine Learning deben superar estas métricas para ser útiles.
    Si un modelo ML no supera el baseline, hay un problema en el pipeline.
    <br><br>
    <b>🎯 Meta de referencia:</b> Intenta superar F1 ≥ {min(1.0, metrics.get('f1', 0) + 0.15):.2f}
    </div>
    """, unsafe_allow_html=True)


# ─── Tab: Conceptos ───────────────────────────────────────────────────────────
def _tab_conceptos() -> None:
    
    st.markdown("### 📖 Conceptos Clave")
    
    concepts = [
        ("🎓 Conjunto de Entrenamiento (Train)", 
         "Los datos que el modelo **aprende** a reconocer patrones. El modelo ajusta sus parámetros internos basándose en estas muestras. Suele ser el 60-80% del total."),
        ("🔍 Conjunto de Validación (Val)", 
         "Datos que **NO** se usaron en entrenamiento, usados para ajustar hiperparámetros y comparar modelos. Ayuda a detectar overfitting sin contaminar el test."),
        ("🧪 Conjunto de Prueba (Test)", 
         "El 'examen final'. El modelo nunca lo ha visto. Evalúa el rendimiento real en producción. **NUNCA** uses el test para tomar decisiones de diseño."),
        ("⚠️ Data Leakage (Fuga de Datos)", 
         "Cuando información del futuro o de los datos de prueba 'contamina' el entrenamiento. Causa métricas artificialmente altas que no se replican en producción."),
        ("🔄 Estratificación", 
         "Técnica que garantiza que cada split (train/val/test) mantenga las mismas proporciones de clases que el dataset original. Evita sesgos por desbalance."),
        ("📊 Modelo Baseline", 
         "Modelo simple de referencia (ej: predecir siempre la clase más frecuente). Es el umbral mínimo que los modelos avanzados deben superar para ser útiles."),
    ]
    
    for title, desc in concepts:
        with st.expander(title):
            st.markdown(f'<div class="insight-box">{desc}</div>', unsafe_allow_html=True)
