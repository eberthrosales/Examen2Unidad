"""
classification.py
=================
Módulo de clasificación supervisada.
Árbol de Decisión + Random Forest con evaluación automática.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import time

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize

from modules.visualization import (
    feature_importance_chart, apply_layout, PALETTE_PRIMARY
)
from modules.interpretation import interpret_classification_metrics


# ─── Sección Principal ────────────────────────────────────────────────────────
def render_classification(df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🌳</div>
        <div>
            <div class="section-title">Clasificación Supervisada</div>
            <div class="section-subtitle">Árbol de Decisión y Random Forest con evaluación automática</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.get('split_done', False):
        st.warning("⚠️ Completa la **Partición de Datos** (sección anterior) primero.")
        return
    
    if st.session_state.get('y_train') is None:
        st.warning("⚠️ No se seleccionó una variable objetivo. Ve a **Preprocesamiento** y selecciona una.")
        return
    
    tabs = st.tabs([
        "⚙️ Configuración", "🌲 Árbol de Decisión",
        "🌳 Random Forest", "📊 Resultados"
    ])
    
    with tabs[0]: _tab_config_clf(df, meta)
    with tabs[1]: _tab_decision_tree(df, meta)
    with tabs[2]: _tab_random_forest(df, meta)
    with tabs[3]: _tab_resultados_clf(df, meta)


# ─── Tab: Configuración ───────────────────────────────────────────────────────
def _tab_config_clf(df: pd.DataFrame, meta: Dict) -> None:
    
    X_train = st.session_state.get('X_train')
    y_train = st.session_state.get('y_train')
    feature_names = st.session_state.get('feature_names', [])
    target_col = st.session_state.get('target_col', 'Target')
    classes = st.session_state.get('target_classes', [])
    
    n_classes = len(np.unique(y_train)) if y_train is not None else 0
    
    # Info general
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{X_train.shape[0]:,}</div>
            <div class="metric-label">Muestras de Train</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{X_train.shape[1]}</div>
            <div class="metric-label">Features</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{n_classes}</div>
            <div class="metric-label">Clases</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown(f"**Variable objetivo:** `{target_col}` | **Clases:** {classes[:10]}")
    
    # Validación cruzada
    st.markdown("#### 🔄 Configuración de Validación Cruzada")
    st.slider("Número de folds (K-Fold):", 3, 10, 5, key='clf_cv_folds',
               help="Más folds = estimación más robusta pero más tiempo.")
    st.selectbox(
        "Métrica principal:", 
        ['f1_weighted', 'accuracy', 'roc_auc_ovr_weighted', 'precision_weighted', 'recall_weighted'],
        key='clf_scoring'
    )
    n_classes = len(np.unique(y_train)) if y_train is not None else 0
    st.session_state['n_classes_clf'] = n_classes
    
    if n_classes == 2:
        avg = 'binary'
    else:
        avg = 'weighted'
    st.session_state['clf_avg'] = avg
    
    st.markdown(f"**Estrategia de promedio:** `{avg}` ({'binario' if avg == 'binary' else 'ponderado para multiclase'})")


# ─── Tab: Árbol de Decisión ───────────────────────────────────────────────────
def _tab_decision_tree(df: pd.DataFrame, meta: Dict) -> None:
    
    X_train = st.session_state.get('X_train')
    y_train = st.session_state.get('y_train')
    X_val   = st.session_state.get('X_val')
    y_val   = st.session_state.get('y_val')
    
    st.markdown("#### 🌲 Árbol de Decisión (Decision Tree Classifier)")
    
    c1, c2 = st.columns(2)
    with c1:
        max_depth = st.slider("Profundidad máxima:", 1, 20, 5, key='dt_max_depth',
                              help="Controla la complejidad. Valores altos → overfitting.")
        min_samples_split = st.slider("Min muestras para dividir:", 2, 50, 10, key='dt_min_split')
        min_samples_leaf = st.slider("Min muestras en hoja:", 1, 30, 5, key='dt_min_leaf')
    with c2:
        criterion = st.selectbox("Criterio:", ['gini', 'entropy', 'log_loss'], key='dt_criterion',
                                 help="gini: Índice Gini. entropy: Ganancia de información.")
        class_weight = st.selectbox("Peso de clases:", ['None', 'balanced'], key='dt_class_weight')
        max_features = st.selectbox("Max features:", ['sqrt', 'log2', 'None'], key='dt_max_feat')
    
    if st.button("🌲 Entrenar Árbol de Decisión", key='btn_dt', type='primary'):
        with st.spinner("Entrenando Árbol de Decisión..."):
            start = time.time()
            
            cw = None if class_weight == 'None' else class_weight
            mf = None if max_features == 'None' else max_features
            
            dt = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                class_weight=cw,
                max_features=mf,
                random_state=42,
            )
            dt.fit(X_train, y_train)
            train_time = round(time.time() - start, 3)
            
            # Métricas
            metrics = _evaluate_model(dt, X_train, y_train, X_val, y_val)
            metrics['train_time'] = train_time
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=st.session_state.get('clf_cv_folds', 5), 
                                  shuffle=True, random_state=42)
            cv_scores = cross_val_score(dt, X_train, y_train, 
                                         cv=cv, scoring='f1_weighted')
            metrics['cv_mean'] = round(float(cv_scores.mean()), 4)
            metrics['cv_std']  = round(float(cv_scores.std()), 4)
            
            st.session_state['dt_model']   = dt
            st.session_state['dt_metrics'] = metrics
            st.session_state['dt_done']    = True
            
            # Guardar para comparación
            _save_model_results('Árbol de Decisión', metrics, dt)
            
            st.success(f"✅ Árbol entrenado en {train_time}s | CV F1: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    
    if st.session_state.get('dt_done', False):
        dt = st.session_state.get('dt_model')
        metrics = st.session_state.get('dt_metrics', {})
        
        _display_model_metrics(metrics, "Árbol de Decisión")
        
        # Feature importance
        feature_names = st.session_state.get('feature_names', [f'f{i}' for i in range(X_train.shape[1])])
        importances = dt.feature_importances_
        
        # Top 20 features
        top_n = min(20, len(feature_names))
        top_idx = np.argsort(importances)[-top_n:]
        
        st.plotly_chart(
            feature_importance_chart(
                [feature_names[i] for i in top_idx],
                importances[top_idx],
                f'Importancia de Variables — Árbol de Decisión (Top {top_n})'
            ),
            use_container_width=True
        )
        
        # Visualización del árbol
        st.markdown("#### 🌿 Visualización del Árbol")
        max_viz_depth = st.slider("Profundidad a visualizar:", 1, min(max_depth, 5), 
                                   min(3, max_depth), key='dt_viz_depth')
        
        with st.spinner("Generando visualización del árbol..."):
            try:
                from sklearn.tree import plot_tree
                
                classes = st.session_state.get('target_classes', None)
                class_str = [str(c) for c in classes] if classes else None
                
                fig_tree, ax = plt.subplots(figsize=(16, 8))
                fig_tree.patch.set_facecolor('#0F0F1A')
                ax.set_facecolor('#0F0F1A')
                
                plot_tree(
                    dt, ax=ax,
                    max_depth=max_viz_depth,
                    feature_names=[str(f)[:15] for f in feature_names],
                    class_names=class_str,
                    filled=True, rounded=True,
                    fontsize=8,
                    proportion=False,
                )
                plt.title(f"Árbol de Decisión (profundidad={max_viz_depth})", 
                         color='#A78BFA', pad=15, fontsize=13)
                plt.tight_layout()
                
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=130, facecolor='#0F0F1A',
                           bbox_inches='tight')
                buf.seek(0)
                st.image(buf, use_column_width=True)
                plt.close()
            except Exception as e:
                st.warning(f"⚠️ No se pudo visualizar el árbol: {e}")
        
        # Interpretación
        baseline_metrics = st.session_state.get('baseline_metrics', {})
        interp = interpret_classification_metrics(metrics, "Árbol de Decisión", baseline_metrics)
        st.markdown(f'<div class="insight-box">{interp}</div>', unsafe_allow_html=True)


# ─── Tab: Random Forest ───────────────────────────────────────────────────────
def _tab_random_forest(df: pd.DataFrame, meta: Dict) -> None:
    
    X_train = st.session_state.get('X_train')
    y_train = st.session_state.get('y_train')
    X_val   = st.session_state.get('X_val')
    y_val   = st.session_state.get('y_val')
    
    st.markdown("#### 🌳 Random Forest Classifier")
    
    c1, c2 = st.columns(2)
    with c1:
        n_estimators = st.slider("Número de árboles:", 10, 500, 100, 10, key='rf_n_est',
                                  help="Más árboles = mejor rendimiento pero más tiempo.")
        max_depth_rf = st.slider("Profundidad máxima:", 1, 30, 10, key='rf_max_depth')
        min_samples_split_rf = st.slider("Min muestras split:", 2, 50, 5, key='rf_min_split')
    with c2:
        max_features_rf = st.selectbox("Max features por árbol:", ['sqrt', 'log2', 'None'], 
                                        key='rf_max_feat')
        class_weight_rf = st.selectbox("Peso de clases:", ['None', 'balanced', 'balanced_subsample'], 
                                        key='rf_class_weight')
        bootstrap = st.checkbox("Bootstrap sampling", value=True, key='rf_bootstrap')
    
    if st.button("🌳 Entrenar Random Forest", key='btn_rf', type='primary'):
        with st.spinner(f"Entrenando Random Forest ({n_estimators} árboles)..."):
            start = time.time()
            
            cw = None if class_weight_rf == 'None' else class_weight_rf
            mf = None if max_features_rf == 'None' else max_features_rf
            
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth_rf,
                min_samples_split=min_samples_split_rf,
                max_features=mf,
                class_weight=cw,
                bootstrap=bootstrap,
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            train_time = round(time.time() - start, 3)
            
            metrics = _evaluate_model(rf, X_train, y_train, X_val, y_val)
            metrics['train_time'] = train_time
            
            # CV
            cv = StratifiedKFold(n_splits=st.session_state.get('clf_cv_folds', 5),
                                  shuffle=True, random_state=42)
            cv_scores = cross_val_score(rf, X_train, y_train,
                                         cv=cv, scoring='f1_weighted')
            metrics['cv_mean'] = round(float(cv_scores.mean()), 4)
            metrics['cv_std']  = round(float(cv_scores.std()), 4)
            
            st.session_state['rf_model']   = rf
            st.session_state['rf_metrics'] = metrics
            st.session_state['rf_done']    = True
            
            _save_model_results('Random Forest', metrics, rf)
            
            st.success(f"✅ Random Forest entrenado en {train_time}s | CV F1: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    
    if st.session_state.get('rf_done', False):
        rf = st.session_state.get('rf_model')
        metrics = st.session_state.get('rf_metrics', {})
        
        _display_model_metrics(metrics, "Random Forest")
        
        # Feature importance
        feature_names = st.session_state.get('feature_names', [f'f{i}' for i in range(X_train.shape[1])])
        importances = rf.feature_importances_
        
        top_n = min(20, len(feature_names))
        top_idx = np.argsort(importances)[-top_n:]
        
        st.plotly_chart(
            feature_importance_chart(
                [feature_names[i] for i in top_idx],
                importances[top_idx],
                f'Importancia de Variables — Random Forest (Top {top_n})'
            ),
            use_container_width=True
        )
        
        # OOB Score
        if bootstrap:
            rf_oob = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth_rf,
                oob_score=True,
                random_state=42, n_jobs=-1
            )
            try:
                rf_oob.fit(X_train, y_train)
                st.markdown(f"""
                <div class="insight-box">
                📊 <b>OOB Score (Out-of-Bag):</b> {rf_oob.oob_score_:.4f}<br>
                El OOB score es una estimación interna del error de generalización usando muestras 
                que no participaron en cada árbol. Similar a validación cruzada pero más eficiente.
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass
        
        # Interpretación
        baseline_metrics = st.session_state.get('baseline_metrics', {})
        interp = interpret_classification_metrics(metrics, "Random Forest", baseline_metrics)
        st.markdown(f'<div class="insight-box">{interp}</div>', unsafe_allow_html=True)


# ─── Tab: Resultados ──────────────────────────────────────────────────────────
def _tab_resultados_clf(df: pd.DataFrame, meta: Dict) -> None:
    
    st.markdown("### 📊 Comparación de Resultados")
    
    if not (st.session_state.get('dt_done') or st.session_state.get('rf_done')):
        st.info("⚙️ Entrena al menos un modelo primero.")
        return
    
    results = st.session_state.get('model_results_df')
    if results is None:
        st.info("No hay resultados guardados.")
        return
    
    st.dataframe(results, use_container_width=True, hide_index=True)
    
    # Gráfico comparativo
    if len(results) >= 2:
        metric_sel = st.selectbox(
            "Métrica a comparar:", 
            ['F1-Score', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC'],
            key='clf_compare_metric'
        )
        
        from modules.visualization import model_comparison_bar
        st.plotly_chart(model_comparison_bar(results, metric=metric_sel), use_container_width=True)
    
    # Cross-validation comparison
    cv_data = []
    for model_name in ['Baseline', 'Árbol de Decisión', 'Random Forest']:
        if model_name == 'Baseline':
            m = st.session_state.get('baseline_metrics', {})
            if m:
                cv_data.append({'Modelo': model_name, 'CV F1 Mean': m.get('f1', 0), 'CV F1 Std': 0})
        elif model_name == 'Árbol de Decisión':
            m = st.session_state.get('dt_metrics', {})
            if m:
                cv_data.append({'Modelo': model_name, 
                               'CV F1 Mean': m.get('cv_mean', m.get('f1', 0)),
                               'CV F1 Std': m.get('cv_std', 0)})
        elif model_name == 'Random Forest':
            m = st.session_state.get('rf_metrics', {})
            if m:
                cv_data.append({'Modelo': model_name,
                               'CV F1 Mean': m.get('cv_mean', m.get('f1', 0)),
                               'CV F1 Std': m.get('cv_std', 0)})
    
    if cv_data:
        cv_df = pd.DataFrame(cv_data)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cv_df['Modelo'],
            y=cv_df['CV F1 Mean'],
            error_y=dict(type='data', array=cv_df['CV F1 Std'], visible=True,
                        color='rgba(255,255,255,0.5)'),
            marker_color=PALETTE_PRIMARY[:len(cv_df)],
            text=[f"{v:.3f} ± {s:.3f}" for v, s in zip(cv_df['CV F1 Mean'], cv_df['CV F1 Std'])],
            textposition='outside',
        ))
        apply_layout(fig, 'Validación Cruzada — F1-Score (Media ± Desviación)', height=400)
        st.plotly_chart(fig, use_container_width=True)


# ─── Funciones Auxiliares ─────────────────────────────────────────────────────
def _evaluate_model(model, X_train, y_train, X_val, y_val) -> Dict:
    """Evalúa un modelo en train y validación."""
    avg = st.session_state.get('clf_avg', 'weighted')
    n_classes = st.session_state.get('n_classes_clf', 2)
    
    y_pred_val = model.predict(X_val)
    
    metrics = {
        'accuracy':        round(float(accuracy_score(y_val, y_pred_val)), 4),
        'precision':       round(float(precision_score(y_val, y_pred_val, average=avg, zero_division=0)), 4),
        'recall':          round(float(recall_score(y_val, y_pred_val, average=avg, zero_division=0)), 4),
        'f1':              round(float(f1_score(y_val, y_pred_val, average=avg, zero_division=0)), 4),
        'train_accuracy':  round(float(accuracy_score(y_train, model.predict(X_train))), 4),
    }
    
    # AUC
    try:
        if n_classes == 2:
            y_prob = model.predict_proba(X_val)[:, 1]
            metrics['auc'] = round(float(roc_auc_score(y_val, y_prob)), 4)
        else:
            y_prob = model.predict_proba(X_val)
            classes = sorted(np.unique(y_train))
            y_bin = label_binarize(y_val, classes=classes)
            metrics['auc'] = round(float(roc_auc_score(y_bin, y_prob, 
                                                         multi_class='ovr', average='weighted')), 4)
    except Exception:
        metrics['auc'] = 0.5
    
    # Overfitting check
    metrics['overfit_gap'] = round(metrics['train_accuracy'] - metrics['accuracy'], 4)
    
    # Guardar predicciones y probabilidades para eval
    metrics['y_pred'] = y_pred_val
    try:
        metrics['y_prob'] = model.predict_proba(X_val)
    except Exception:
        metrics['y_prob'] = None
    
    return metrics


def _display_model_metrics(metrics: Dict, model_name: str) -> None:
    """Muestra métricas del modelo con colores."""
    
    st.markdown(f"#### 📈 Métricas — {model_name}")
    
    # CV info
    if 'cv_mean' in metrics:
        st.markdown(f"**Validación Cruzada F1:** `{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}`")
    
    cols = st.columns(5)
    metric_list = [
        ("Accuracy",  metrics.get('accuracy', 0)),
        ("Precision", metrics.get('precision', 0)),
        ("Recall",    metrics.get('recall', 0)),
        ("F1-Score",  metrics.get('f1', 0)),
        ("AUC-ROC",   metrics.get('auc', 0)),
    ]
    
    for i, (name, val) in enumerate(metric_list):
        with cols[i]:
            color = '#10B981' if val > 0.7 else '#F59E0B' if val > 0.5 else '#EF4444'
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{color}">{val:.3f}</div>
                <div class="metric-label">{name}</div>
            </div>""", unsafe_allow_html=True)
    
    # Overfitting warning
    overfit = metrics.get('overfit_gap', 0)
    train_acc = metrics.get('train_accuracy', 0)
    if overfit > 0.15:
        st.markdown(f"""
        <div class="insight-box">
        ⚠️ <b>Posible Overfitting detectado</b>: Train Accuracy = {train_acc:.3f} vs Val Accuracy = {metrics.get('accuracy', 0):.3f} 
        (diferencia = {overfit:.3f}). Considera reducir la profundidad máxima o aumentar la regularización.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"⏱️ Tiempo de entrenamiento: `{metrics.get('train_time', 0)}s`")


def _save_model_results(model_name: str, metrics: Dict, model=None) -> None:
    """Guarda resultados para comparación."""
    
    models_dict = st.session_state.get('trained_models', {})
    models_dict[model_name] = {'model': model, 'metrics': metrics}
    st.session_state['trained_models'] = models_dict
    
    # Actualizar DataFrame de comparación
    results = []
    for name, info in models_dict.items():
        m = info.get('metrics', {})
        results.append({
            'Modelo':         name,
            'Accuracy':       m.get('accuracy', 0),
            'Precision':      m.get('precision', 0),
            'Recall':         m.get('recall', 0),
            'F1-Score':       m.get('f1', 0),
            'AUC-ROC':        m.get('auc', 0),
            'Tiempo (s)':     m.get('train_time', '-'),
        })
    
    st.session_state['model_results_df'] = pd.DataFrame(results)
