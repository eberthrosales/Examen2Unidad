"""
evaluation.py
=============
Evaluación completa: Matriz de Confusión, Curvas ROC y Comparación de Modelos.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from modules.visualization import (
    confusion_matrix_plot, roc_curve_plot, model_comparison_bar,
    apply_layout, PALETTE_PRIMARY
)
from modules.interpretation import interpret_confusion_matrix, interpret_model_comparison


# ─── Sección Principal ────────────────────────────────────────────────────────
def render_evaluation(df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📉</div>
        <div>
            <div class="section-title">Evaluación y Comparación de Modelos</div>
            <div class="section-subtitle">Matriz de Confusión, Curvas ROC y Ranking de Modelos</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not (st.session_state.get('dt_done') or st.session_state.get('rf_done') or 
            st.session_state.get('baseline_metrics')):
        st.warning("⚠️ Entrena al menos un modelo en la sección de **Clasificación** primero.")
        return
    
    tabs = st.tabs([
        "🟥 Matriz de Confusión", "📈 Curvas ROC", 
        "🏆 Comparación", "🧪 Evaluación Final (Test)"
    ])
    
    with tabs[0]: _tab_confusion(df, meta)
    with tabs[1]: _tab_roc(df, meta)
    with tabs[2]: _tab_comparison(df, meta)
    with tabs[3]: _tab_test_eval(df, meta)


# ─── Tab: Matriz de Confusión ─────────────────────────────────────────────────
def _tab_confusion(df: pd.DataFrame, meta: Dict) -> None:
    
    st.markdown("### 🟥 Matriz de Confusión")
    
    model_options = _get_available_models()
    if not model_options:
        st.info("No hay modelos entrenados.")
        return
    
    model_sel = st.selectbox("Selecciona modelo:", model_options, key='cm_model_sel')
    
    X_val  = st.session_state.get('X_val')
    y_val  = st.session_state.get('y_val')
    classes = st.session_state.get('target_classes', [])
    
    model_obj, metrics = _get_model_and_metrics(model_sel)
    
    if model_obj is None:
        st.warning("⚠️ Modelo no disponible. Re-entrena desde la sección de Clasificación.")
        return
    
    # Calcular confusion matrix
    y_pred = model_obj.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    
    # Etiquetas
    unique_labels = sorted(np.unique(y_val))
    if classes and len(classes) == len(unique_labels):
        label_names = [str(classes[int(l)]) if int(l) < len(classes) else str(l) 
                      for l in unique_labels]
    else:
        label_names = [str(l) for l in unique_labels]
    
    # Heatmap
    st.plotly_chart(confusion_matrix_plot(cm, label_names), use_container_width=True)
    
    # Interpretación automática
    st.markdown("#### 💬 Interpretación Automática")
    interp = interpret_confusion_matrix(cm, label_names)
    st.markdown(interp)
    
    # Classification report
    with st.expander("📋 Reporte Completo (Classification Report)"):
        report = classification_report(y_val, y_pred, 
                                       target_names=label_names,
                                       output_dict=True,
                                       zero_division=0)
        report_df = pd.DataFrame(report).T.round(4)
        st.dataframe(report_df, use_container_width=True)
    
    # Errores más frecuentes (para multiclase)
    if cm.shape[0] > 2:
        st.markdown("#### ❌ Errores más Frecuentes")
        errors = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    errors.append({
                        'Real': label_names[i] if i < len(label_names) else str(i),
                        'Predicho': label_names[j] if j < len(label_names) else str(j),
                        'Conteo': cm[i, j],
                    })
        
        if errors:
            err_df = pd.DataFrame(errors).sort_values('Conteo', ascending=False).head(10)
            st.dataframe(err_df, use_container_width=True, hide_index=True)


# ─── Tab: Curvas ROC ──────────────────────────────────────────────────────────
def _tab_roc(df: pd.DataFrame, meta: Dict) -> None:
    
    st.markdown("### 📈 Curvas ROC")
    
    X_val  = st.session_state.get('X_val')
    y_val  = st.session_state.get('y_val')
    n_classes = st.session_state.get('n_classes_clf', 2)
    classes = st.session_state.get('target_classes', [])
    
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    
    model_options = _get_available_models()
    models_sel = st.multiselect(
        "Modelos a comparar:", model_options,
        default=model_options[:3],
        key='roc_models'
    )
    
    for model_name in models_sel:
        model_obj, _ = _get_model_and_metrics(model_name)
        if model_obj is None:
            continue
        
        try:
            if n_classes == 2:
                y_prob = model_obj.predict_proba(X_val)[:, 1]
                fpr, tpr, _ = roc_curve(y_val, y_prob)
                auc_val = auc(fpr, tpr)
                fpr_dict[model_name] = fpr
                tpr_dict[model_name] = tpr
                auc_dict[model_name] = auc_val
            else:
                # OvR para multiclase — usar macro average
                unique_classes = sorted(np.unique(y_val))
                y_prob = model_obj.predict_proba(X_val)
                
                all_fpr = np.unique(np.concatenate([
                    roc_curve(label_binarize(y_val, classes=unique_classes)[:, i],
                             y_prob[:, i])[0] for i in range(len(unique_classes))
                ]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(len(unique_classes)):
                    y_bin = label_binarize(y_val, classes=unique_classes)[:, i]
                    fpr_i, tpr_i, _ = roc_curve(y_bin, y_prob[:, i])
                    mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
                mean_tpr /= len(unique_classes)
                
                auc_val = auc(all_fpr, mean_tpr)
                fpr_dict[model_name] = all_fpr
                tpr_dict[model_name] = mean_tpr
                auc_dict[model_name] = auc_val
        except Exception as e:
            st.warning(f"⚠️ No se pudo calcular ROC para {model_name}: {e}")
    
    if fpr_dict:
        st.plotly_chart(roc_curve_plot(fpr_dict, tpr_dict, auc_dict), use_container_width=True)
        
        # Tabla AUC
        auc_df = pd.DataFrame([
            {'Modelo': k, 'AUC-ROC': round(v, 4)} for k, v in auc_dict.items()
        ]).sort_values('AUC-ROC', ascending=False)
        st.dataframe(auc_df, use_container_width=True, hide_index=True)
        
        # Interpretación
        best = auc_df.iloc[0]
        auc_val = best['AUC-ROC']
        if auc_val > 0.9:
            quality = "excelente"
        elif auc_val > 0.8:
            quality = "muy buena"
        elif auc_val > 0.7:
            quality = "buena"
        elif auc_val > 0.6:
            quality = "moderada"
        else:
            quality = "pobre"
        
        st.markdown(f"""
        <div class="insight-box">
        📊 <b>Interpretación de la Curva ROC</b><br><br>
        El mejor modelo es <b>{best['Modelo']}</b> con AUC = <b>{auc_val:.3f}</b> 
        (discriminación <b>{quality}</b>).<br><br>
        • AUC = 1.0: clasificador perfecto<br>
        • AUC = 0.5: clasificador aleatorio (línea diagonal)<br>
        • AUC < 0.5: peor que aleatoriedad (posible error en el target)<br><br>
        La curva ROC muestra el trade-off entre <b>True Positive Rate</b> (sensibilidad) y 
        <b>False Positive Rate</b> para todos los umbrales de decisión posibles.
        </div>
        """, unsafe_allow_html=True)
    
    # Multiclase: ROC por clase
    if n_classes > 2:
        with st.expander("🔍 Curvas ROC por Clase Individual"):
            model_sel_roc = st.selectbox("Modelo:", model_options, key='roc_per_class')
            model_obj, _ = _get_model_and_metrics(model_sel_roc)
            
            if model_obj is not None:
                try:
                    unique_classes = sorted(np.unique(y_val))
                    y_prob = model_obj.predict_proba(X_val)
                    
                    fpr_per, tpr_per, auc_per = {}, {}, {}
                    for i, cls in enumerate(unique_classes):
                        y_bin = (y_val == cls).astype(int)
                        fpr_c, tpr_c, _ = roc_curve(y_bin, y_prob[:, i])
                        label = str(classes[int(cls)]) if classes and int(cls) < len(classes) else str(cls)
                        fpr_per[label] = fpr_c
                        tpr_per[label] = tpr_c
                        auc_per[label] = auc(fpr_c, tpr_c)
                    
                    st.plotly_chart(roc_curve_plot(fpr_per, tpr_per, auc_per), 
                                   use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ {e}")


# ─── Tab: Comparación ─────────────────────────────────────────────────────────
def _tab_comparison(df: pd.DataFrame, meta: Dict) -> None:
    
    st.markdown("### 🏆 Comparación de Modelos")
    
    results_df = st.session_state.get('model_results_df')
    
    if results_df is None or results_df.empty:
        st.info("No hay modelos entrenados para comparar.")
        return
    
    # Ranking
    results_sorted = results_df.sort_values('F1-Score', ascending=False).copy()
    results_sorted.insert(0, 'Ranking', range(1, len(results_sorted)+1))
    
    # Emojis de ranking
    emojis = ['🥇', '🥈', '🥉'] + ['  ' for _ in range(len(results_sorted)-3)]
    results_sorted[''] = emojis[:len(results_sorted)]
    
    # Resaltar mejor modelo
    def highlight_best(s):
        styles = []
        for i, v in enumerate(s):
            if i == 0 and s.name in ['F1-Score', 'Accuracy', 'AUC-ROC']:
                styles.append('background-color: rgba(16,185,129,0.2); font-weight: bold')
            else:
                styles.append('')
        return styles
    
    st.dataframe(
        results_sorted[['Ranking', '', 'Modelo', 'Accuracy', 'Precision', 
                         'Recall', 'F1-Score', 'AUC-ROC', 'Tiempo (s)']],
        use_container_width=True,
        hide_index=True,
    )
    
    # Gráficos comparativos
    metric_opts = ['F1-Score', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']
    
    # Radar chart comparativo
    st.markdown("#### 🕸️ Radar Comparativo")
    models_for_radar = results_df['Modelo'].tolist()[:4]  # max 4 modelos
    
    fig_radar = go.Figure()
    for i, model_name in enumerate(models_for_radar):
        row = results_df[results_df['Modelo'] == model_name].iloc[0]
        values = [row.get(m, 0) for m in metric_opts]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=metric_opts + [metric_opts[0]],
            name=model_name,
            fill='toself',
            fillcolor=f'rgba({",".join(str(int(c,16)) for c in [PALETTE_PRIMARY[i][1:3], PALETTE_PRIMARY[i][3:5], PALETTE_PRIMARY[i][5:7]])}, 0.1)',
            line=dict(color=PALETTE_PRIMARY[i], width=2),
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                           gridcolor='rgba(148,163,184,0.2)'),
            angularaxis=dict(gridcolor='rgba(148,163,184,0.2)'),
            bgcolor='rgba(26,26,46,0.5)',
        ),
    )
    apply_layout(fig_radar, 'Comparación Multi-Métrica', height=440)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Barras por métrica
    metric_cmp = st.selectbox("Ver gráfico de barras por:", metric_opts, key='eval_bar_metric')
    st.plotly_chart(model_comparison_bar(results_df, metric=metric_cmp), use_container_width=True)
    
    # Interpretación
    st.markdown("#### 💬 Análisis del Ranking")
    st.markdown(interpret_model_comparison(results_df))


# ─── Tab: Evaluación Final en Test ────────────────────────────────────────────
def _tab_test_eval(df: pd.DataFrame, meta: Dict) -> None:
    
    st.markdown("### 🧪 Evaluación Final en Conjunto de Test")
    
    st.markdown("""
    <div class="insight-box">
    ⚠️ <b>IMPORTANTE</b>: El conjunto de test es el "examen final". Solo debe usarse 
    <b>una vez</b> para reportar el rendimiento final del modelo seleccionado. 
    Evaluar múltiples veces en test puede causar <b>data leakage optimista</b>.
    </div>
    """, unsafe_allow_html=True)
    
    X_test = st.session_state.get('X_test')
    y_test = st.session_state.get('y_test')
    
    if X_test is None or y_test is None:
        st.warning("⚠️ No hay conjunto de test disponible. Realiza la partición primero.")
        return
    
    model_options = _get_available_models()
    if not model_options:
        st.info("No hay modelos entrenados.")
        return
    
    best_model_default = None
    results_df = st.session_state.get('model_results_df')
    if results_df is not None and not results_df.empty:
        best_model_default = results_df.sort_values('F1-Score', ascending=False).iloc[0]['Modelo']
    
    default_idx = model_options.index(best_model_default) if best_model_default in model_options else 0
    
    model_sel = st.selectbox(
        "Selecciona el modelo final a evaluar en test:",
        model_options,
        index=default_idx,
        key='test_model_sel'
    )
    
    st.warning(f"⚠️ Vas a evaluar **{model_sel}** en el conjunto de test. Esta acción no debe repetirse.")
    
    if st.button("🧪 Evaluar en Test", key='btn_test_eval', type='primary'):
        model_obj, val_metrics = _get_model_and_metrics(model_sel)
        
        if model_obj is None:
            st.error("❌ Modelo no disponible.")
            return
        
        avg = st.session_state.get('clf_avg', 'weighted')
        n_classes = st.session_state.get('n_classes_clf', 2)
        classes = st.session_state.get('target_classes', [])
        
        y_pred_test = model_obj.predict(X_test)
        
        test_metrics = {
            'accuracy':  round(float(accuracy_score(y_test, y_pred_test)), 4),
            'precision': round(float(precision_score(y_test, y_pred_test, average=avg, zero_division=0)), 4),
            'recall':    round(float(recall_score(y_test, y_pred_test, average=avg, zero_division=0)), 4),
            'f1':        round(float(f1_score(y_test, y_pred_test, average=avg, zero_division=0)), 4),
        }
        
        try:
            if n_classes == 2:
                y_prob_test = model_obj.predict_proba(X_test)[:, 1]
                test_metrics['auc'] = round(float(roc_auc_score(y_test, y_prob_test)), 4)
            else:
                y_prob_test = model_obj.predict_proba(X_test)
                unique_c = sorted(np.unique(y_test))
                y_bin = label_binarize(y_test, classes=unique_c)
                test_metrics['auc'] = round(float(roc_auc_score(y_bin, y_prob_test,
                                                                 multi_class='ovr', average='weighted')), 4)
        except Exception:
            test_metrics['auc'] = 0.5
        
        st.session_state['test_metrics'] = test_metrics
        st.session_state['best_model_final'] = model_sel
        
        # Display
        st.markdown(f"#### 📊 Resultados Finales en Test — {model_sel}")
        
        metric_list = [
            ("Accuracy",  test_metrics['accuracy']),
            ("Precision", test_metrics['precision']),
            ("Recall",    test_metrics['recall']),
            ("F1-Score",  test_metrics['f1']),
            ("AUC-ROC",   test_metrics['auc']),
        ]
        
        cols = st.columns(5)
        for i, (name, val) in enumerate(metric_list):
            val_m = val_metrics.get(name.lower().replace('-', ''), 0) if val_metrics else 0
            delta = val - val_m
            color = '#10B981' if val > 0.7 else '#F59E0B' if val > 0.5 else '#EF4444'
            delta_class = 'positive' if delta >= 0 else 'negative'
            
            with cols[i]:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:{color}">{val:.3f}</div>
                    <div class="metric-label">{name}</div>
                    <div class="metric-delta {delta_class}">{'↑' if delta >= 0 else '↓'} {abs(delta):.3f} vs Val</div>
                </div>""", unsafe_allow_html=True)
        
        # Confusion matrix en test
        unique_labels = sorted(np.unique(y_test))
        if classes and len(classes) >= len(unique_labels):
            label_names = [str(classes[int(l)]) if int(l) < len(classes) else str(l) 
                          for l in unique_labels]
        else:
            label_names = [str(l) for l in unique_labels]
        
        cm_test = confusion_matrix(y_test, y_pred_test)
        st.plotly_chart(confusion_matrix_plot(cm_test, label_names), use_container_width=True)
        
        # Guardar para export
        st.session_state['best_model_metrics'] = {**test_metrics, 'model_name': model_sel}


# ─── Funciones Auxiliares ─────────────────────────────────────────────────────
def _get_available_models() -> List[str]:
    """Retorna lista de modelos entrenados disponibles."""
    models = []
    if st.session_state.get('baseline_metrics'):
        models.append('Baseline')
    if st.session_state.get('dt_done'):
        models.append('Árbol de Decisión')
    if st.session_state.get('rf_done'):
        models.append('Random Forest')
    return models


def _get_model_and_metrics(model_name: str):
    """Retorna (model_object, metrics_dict) para un modelo dado."""
    trained = st.session_state.get('trained_models', {})
    
    if model_name in trained:
        info = trained[model_name]
        return info.get('model'), info.get('metrics', {})
    
    # Fallback directo
    if model_name == 'Baseline':
        m = st.session_state.get('baseline_metrics', {})
        return m.get('model'), m
    elif model_name == 'Árbol de Decisión':
        return st.session_state.get('dt_model'), st.session_state.get('dt_metrics', {})
    elif model_name == 'Random Forest':
        return st.session_state.get('rf_model'), st.session_state.get('rf_metrics', {})
    
    return None, {}
