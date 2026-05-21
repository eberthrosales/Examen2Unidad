"""
interpretation.py
=================
Generación automática de interpretaciones en lenguaje natural.
Sin nombres de columnas hardcodeados.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


# ─── Interpretación de Métricas de Clasificación ─────────────────────────────
def interpret_classification_metrics(metrics: Dict, model_name: str,
                                      baseline_metrics: Dict = None) -> str:
    """Genera interpretación automática de métricas de clasificación."""
    acc  = metrics.get('accuracy', 0)
    prec = metrics.get('precision', 0)
    rec  = metrics.get('recall', 0)
    f1   = metrics.get('f1', 0)
    auc  = metrics.get('auc', 0)
    
    # Evaluación de rendimiento
    if f1 >= 0.90:
        performance = "excelente"
        emoji = "🏆"
    elif f1 >= 0.80:
        performance = "muy bueno"
        emoji = "✅"
    elif f1 >= 0.70:
        performance = "bueno"
        emoji = "👍"
    elif f1 >= 0.60:
        performance = "moderado"
        emoji = "⚠️"
    else:
        performance = "mejorable"
        emoji = "❌"
    
    # Comparación con baseline
    baseline_text = ""
    if baseline_metrics:
        b_f1 = baseline_metrics.get('f1', 0)
        improvement = (f1 - b_f1) / max(b_f1, 0.001) * 100
        if improvement > 20:
            baseline_text = f" Supera al baseline en **{improvement:.1f}%** en F1-Score."
        elif improvement > 5:
            baseline_text = f" Mejora ligeramente al baseline ({improvement:.1f}% en F1)."
        elif improvement > 0:
            baseline_text = f" Supera marginalmente al baseline (+{improvement:.1f}%)."
        else:
            baseline_text = f" ⚠️ No supera al baseline. Revisa el pipeline y las features."
    
    # Análisis precision vs recall
    pr_text = ""
    if prec > 0 and rec > 0:
        if prec > rec + 0.15:
            pr_text = "El modelo es más **preciso** que sensible: cuando predice positivo, generalmente acierta, pero puede perder casos reales."
        elif rec > prec + 0.15:
            pr_text = "El modelo es más **sensible** que preciso: detecta la mayoría de los positivos, pero genera algunos falsos positivos."
        else:
            pr_text = "El modelo tiene un **buen equilibrio** entre Precision y Recall."
    
    text = f"""
    {emoji} **{model_name}** muestra un rendimiento **{performance}** con F1-Score de **{f1:.3f}**.
    {baseline_text}
    
    - **Accuracy ({acc:.3f})**: {acc*100:.1f}% de las predicciones son correctas.
    - **Precision ({prec:.3f})**: De cada predicción positiva, {prec*100:.1f}% es realmente positiva.
    - **Recall ({rec:.3f})**: El modelo detecta el {rec*100:.1f}% de los positivos reales.
    - **AUC-ROC ({auc:.3f})**: {"Excelente discriminación" if auc > 0.9 else "Buena discriminación" if auc > 0.8 else "Discriminación moderada" if auc > 0.7 else "Discriminación pobre"}.
    
    {pr_text}
    """
    return text


# ─── Interpretación de Matriz de Confusión ───────────────────────────────────
def interpret_confusion_matrix(cm: np.ndarray, class_labels: List) -> str:
    """Interpreta la matriz de confusión en lenguaje simple."""
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        text = f"""
**📊 Análisis de la Matriz de Confusión:**

| Categoría | Valor | Interpretación |
|-----------|-------|----------------|
| ✅ Verdaderos Positivos (TP) | **{tp}** | Casos positivos correctamente identificados |
| ✅ Verdaderos Negativos (TN) | **{tn}** | Casos negativos correctamente identificados |
| ⚠️ Falsos Positivos (FP) | **{fp}** | Negativos incorrectamente clasificados como positivos (Error Tipo I) |
| ⚠️ Falsos Negativos (FN) | **{fn}** | Positivos incorrectamente clasificados como negativos (Error Tipo II) |

**💡 Impacto de los Errores:**
- **Falsos Positivos ({fp}, {fp/total*100:.1f}%)**: El modelo predijo positivo cuando era negativo. 
  Dependiendo del contexto: puede causar tratamientos innecesarios, alertas falsas, etc.
- **Falsos Negativos ({fn}, {fn/total*100:.1f}%)**: El modelo predijo negativo cuando era positivo.
  Generalmente más crítico: casos perdidos, diagnósticos tardíos, etc.

{"⚠️ **Alta tasa de FN**: El modelo pierde muchos casos positivos. Considera ajustar el umbral de decisión o usar técnicas de balanceo." if fn > tp * 0.3 else ""}
{"⚠️ **Alta tasa de FP**: El modelo genera muchas alarmas falsas. Considera aumentar el umbral de decisión." if fp > tn * 0.3 else ""}
        """
    else:
        # Multiclase
        n = cm.shape[0]
        total = cm.sum()
        correct = np.diag(cm).sum()
        
        lines = [f"**📊 Matriz de Confusión Multiclase ({n} clases):**\n"]
        lines.append(f"- Total de predicciones: **{total}**")
        lines.append(f"- Correctas: **{correct}** ({correct/total*100:.1f}%)")
        lines.append(f"- Incorrectas: **{total-correct}** ({(total-correct)/total*100:.1f}%)\n")
        
        lines.append("**🎯 Rendimiento por clase:**")
        for i, label in enumerate(class_labels):
            tp_i = cm[i, i]
            total_real = cm[i, :].sum()
            total_pred = cm[:, i].sum()
            recall_i = tp_i / max(total_real, 1)
            prec_i   = tp_i / max(total_pred, 1)
            lines.append(f"- **{label}**: Recall={recall_i:.2f}, Precision={prec_i:.2f} ({tp_i}/{total_real} correctos)")
        
        text = "\n".join(lines)
    
    return text


# ─── Interpretación de Clusters ───────────────────────────────────────────────
def interpret_clusters(cluster_profiles: pd.DataFrame, n_clusters: int,
                       numeric_cols: List[str], method: str = "K-Means") -> str:
    """Genera perfiles automáticos de cada cluster."""
    
    lines = [f"### 🔍 Interpretación de {n_clusters} Clusters ({method})\n"]
    
    if cluster_profiles is None or cluster_profiles.empty:
        return "No se pudieron generar perfiles de clusters."
    
    global_means = cluster_profiles[numeric_cols].mean() if numeric_cols else None
    
    for cluster_id in sorted(cluster_profiles.index.unique() if 'Cluster' not in cluster_profiles.columns 
                              else cluster_profiles['Cluster'].unique()):
        
        if 'Cluster' in cluster_profiles.columns:
            row = cluster_profiles[cluster_profiles['Cluster'] == cluster_id]
        else:
            row = cluster_profiles.loc[[cluster_id]]
        
        if row.empty:
            continue
        
        n_members = int(row.get('N_miembros', row.get('count', [0])).iloc[0]) if 'N_miembros' in row.columns or 'count' in row.columns else 0
        
        lines.append(f"#### 🟣 Cluster {cluster_id}" + (f" ({n_members:,} miembros)" if n_members > 0 else ""))
        
        # Características distintivas
        distinctions = []
        if global_means is not None:
            for col in numeric_cols[:8]:
                if col in row.columns:
                    val = row[col].iloc[0]
                    glob = global_means.get(col, val)
                    if glob != 0:
                        diff_pct = (val - glob) / abs(glob) * 100
                        if diff_pct > 20:
                            distinctions.append(f"**{col}** alto ({val:.2f} vs media {glob:.2f}, +{diff_pct:.0f}%)")
                        elif diff_pct < -20:
                            distinctions.append(f"**{col}** bajo ({val:.2f} vs media {glob:.2f}, {diff_pct:.0f}%)")
        
        if distinctions:
            lines.append("Características destacadas: " + ", ".join(distinctions[:4]))
        else:
            lines.append("Características similares al promedio general.")
        
        lines.append("")
    
    lines.append("\n**💡 Recomendaciones:**")
    lines.append("- Analiza las variables más diferenciadas entre clusters para entender qué los separa.")
    lines.append("- Cada cluster puede representar un segmento con comportamiento distinto.")
    lines.append("- Usa el perfil de cada cluster para diseñar estrategias diferenciadas.")
    
    return "\n".join(lines)


# ─── Comparación de Modelos ───────────────────────────────────────────────────
def interpret_model_comparison(results_df: pd.DataFrame) -> str:
    """Genera explicación del ranking de modelos."""
    
    if results_df is None or results_df.empty:
        return "No hay modelos para comparar."
    
    best_idx  = results_df['F1-Score'].idxmax()
    best_row  = results_df.loc[best_idx]
    worst_row = results_df.loc[results_df['F1-Score'].idxmin()]
    
    text = f"""
### 🏆 Evaluación Comparativa de Modelos

**Mejor modelo: {best_row['Modelo']}**
- F1-Score: **{best_row['F1-Score']:.3f}**
- Accuracy: **{best_row['Accuracy']:.3f}**
- AUC-ROC: **{best_row['AUC-ROC']:.3f}**

**¿Por qué {best_row['Modelo']}?**
"""
    
    model_name = str(best_row['Modelo'])
    if 'Random Forest' in model_name:
        text += "Random Forest combina múltiples árboles de decisión, reduciendo overfitting y capturando relaciones complejas. Es robusto frente a outliers y variables irrelevantes."
    elif 'Árbol' in model_name or 'Decision Tree' in model_name:
        text += "El Árbol de Decisión ofrece alta interpretabilidad. Sus reglas de decisión son directamente comprensibles por cualquier persona."
    elif 'Baseline' in model_name:
        text += "⚠️ El baseline es el mejor modelo, lo que sugiere problemas en el pipeline. Revisa las features, el target y el preprocesamiento."
    else:
        text += f"El modelo {model_name} encontró patrones relevantes en los datos."
    
    text += f"""

**Ranking completo:**
"""
    for _, row in results_df.sort_values('F1-Score', ascending=False).iterrows():
        is_best = row['Modelo'] == best_row['Modelo']
        text += f"\n{'🥇' if is_best else '  •'} **{row['Modelo']}**: F1={row['F1-Score']:.3f}, Acc={row['Accuracy']:.3f}"
    
    # Recomendaciones
    f1_diff = best_row['F1-Score'] - worst_row['F1-Score']
    text += f"""

**💡 Recomendaciones:**
- {"El mejor modelo destaca notablemente. Úsalo para producción." if f1_diff > 0.1 else "Los modelos tienen rendimiento similar. Considera la interpretabilidad y velocidad."}
- Valida el modelo seleccionado en el conjunto de **test** para confirmar su rendimiento real.
- {"Considera ajuste de hiperparámetros más profundo (Grid Search) para mejorar aún más." if best_row['F1-Score'] < 0.9 else "El rendimiento es muy bueno. Verifica que no haya overfitting."}
"""
    
    return text


# ─── Recomendaciones Generales ────────────────────────────────────────────────
def generate_recommendations(meta: Dict, metrics: Dict = None, 
                               cluster_info: Dict = None) -> List[str]:
    """Genera recomendaciones contextuales basadas en el dataset y resultados."""
    
    recs = []
    
    # Basadas en dataset
    n_rows = meta.get('n_rows', 0)
    null_total = sum(meta.get('null_counts', {}).values())
    
    if n_rows < 500:
        recs.append("📊 **Dataset pequeño**: Usa validación cruzada k-fold en lugar de un único split para obtener estimaciones más robustas.")
    
    if null_total > 0:
        recs.append("🏥 **Valores nulos presentes**: El pipeline de imputación se aplicó automáticamente. Verifica que la estrategia elegida sea apropiada para tu dominio.")
    
    n_cats = len(meta.get('categorical_cols', []))
    if n_cats > 10:
        recs.append("🏷️ **Muchas variables categóricas**: Si usas OHE, el número de features puede aumentar significativamente. Considera Label Encoding o embeddings para reducir dimensionalidad.")
    
    # Basadas en métricas
    if metrics:
        f1 = metrics.get('f1', 0)
        if f1 < 0.6:
            recs.append("⚠️ **Bajo F1-Score**: Considera ingeniería de features, más datos, o modelos más complejos (Gradient Boosting, XGBoost).")
        
        prec = metrics.get('precision', 0)
        rec  = metrics.get('recall', 0)
        if abs(prec - rec) > 0.2:
            recs.append(f"⚖️ **Desbalance Precision/Recall**: Ajusta el umbral de decisión del modelo para balancear errores según las necesidades del negocio.")
    
    # Basadas en clusters
    if cluster_info:
        silhouette = cluster_info.get('silhouette', 0)
        if silhouette < 0.3:
            recs.append("🔍 **Silhouette bajo**: Los clusters no están bien separados. Considera más features, normalización diferente, o un número distinto de clusters.")
    
    if not recs:
        recs.append("✅ El análisis no detectó problemas graves. El pipeline parece estar configurado correctamente.")
    
    return recs


# ─── Resumen Ejecutivo ────────────────────────────────────────────────────────
def generate_executive_summary(meta: Dict, best_model_metrics: Dict = None,
                                cluster_info: Dict = None) -> str:
    """Genera un resumen ejecutivo del análisis completo."""
    
    n_rows = meta.get('n_rows', 0)
    n_cols = meta.get('n_cols', 0)
    n_num  = len(meta.get('numeric_cols', []))
    n_cat  = len(meta.get('categorical_cols', []))
    
    summary = f"""
## 📋 Resumen Ejecutivo del Análisis

### Dataset
- **{n_rows:,}** registros con **{n_cols}** variables ({n_num} numéricas, {n_cat} categóricas)
- Completitud: **{round((1 - sum(meta.get('null_counts', {}).values())/(n_rows*n_cols))*100, 1)}%**
"""
    
    if best_model_metrics:
        model_name = best_model_metrics.get('model_name', 'Mejor modelo')
        f1 = best_model_metrics.get('f1', 0)
        acc = best_model_metrics.get('accuracy', 0)
        perf = "Excelente" if f1 > 0.9 else "Muy bueno" if f1 > 0.8 else "Bueno" if f1 > 0.7 else "Moderado"
        
        summary += f"""
### Clasificación
- Mejor modelo: **{model_name}**
- Rendimiento: **{perf}** (F1={f1:.3f}, Accuracy={acc:.3f})
"""
    
    if cluster_info:
        k = cluster_info.get('k', '?')
        method = cluster_info.get('method', 'K-Means')
        sil = cluster_info.get('silhouette', 0)
        quality = "Buena" if sil > 0.5 else "Moderada" if sil > 0.3 else "Baja"
        
        summary += f"""
### Segmentación
- Método: **{method}** con **{k}** clusters
- Calidad de segmentación: **{quality}** (Silhouette={sil:.3f})
"""
    
    summary += """
### Próximos Pasos Recomendados
1. Validar el modelo en datos de producción reales
2. Monitorear el rendimiento ante concept drift
3. Reentrenar periódicamente con nuevos datos
4. Considerar modelos avanzados si el rendimiento no es suficiente
"""
    
    return summary
