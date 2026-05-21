"""
eda.py
======
Análisis Exploratorio de Datos (EDA) completamente dinámico.
Sin nombres de columnas hardcodeados.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

from modules.visualization import (
    histogram, boxplot, bar_categorical,
    correlation_heatmap, null_heatmap, pairplot_sample,
    scatter_2d, apply_layout, PALETTE_PRIMARY
)
from modules.data_loader import analyze_dataset, get_column_summary


# ─── Sección Principal ────────────────────────────────────────────────────────
def render_eda(df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    """Renderiza el módulo completo de EDA."""
    
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔍</div>
        <div>
            <div class="section-title">Análisis Exploratorio de Datos</div>
            <div class="section-subtitle">Exploración automática del dataset cargado</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs([
        "📊 Resumen", "🔢 Estadísticas", "📈 Distribuciones",
        "🔗 Correlaciones", "🚫 Valores Nulos", "⚠️ Outliers", "🎯 Análisis de Clases"
    ])
    
    with tabs[0]: _tab_resumen(df, meta)
    with tabs[1]: _tab_estadisticas(df, meta)
    with tabs[2]: _tab_distribuciones(df, meta)
    with tabs[3]: _tab_correlaciones(df, meta)
    with tabs[4]: _tab_nulos(df, meta)
    with tabs[5]: _tab_outliers(df, meta)
    with tabs[6]: _tab_clases(df, meta)


# ─── Tab: Resumen ─────────────────────────────────────────────────────────────
def _tab_resumen(df: pd.DataFrame, meta: Dict) -> None:
    # KPIs principales
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "📋", "Filas", f"{meta['n_rows']:,}", None),
        (c2, "📐", "Columnas", str(meta['n_cols']), None),
        (c3, "🔢", "Numéricas", str(len(meta['numeric_cols'])), None),
        (c4, "🏷️", "Categóricas", str(len(meta['categorical_cols'])), None),
    ]
    for col, icon, label, val, delta in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem">{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    
    null_total = sum(meta['null_counts'].values())
    with c5:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:1.5rem">🕳️</div>
            <div class="metric-value">{null_total:,}</div>
            <div class="metric-label">Valores Nulos</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:1.5rem">💾</div>
            <div class="metric-value">{meta['memory_mb']} MB</div>
            <div class="metric-label">Memoria</div>
        </div>""", unsafe_allow_html=True)
    with c7:
        n_dup = df.duplicated().sum()
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:1.5rem">📑</div>
            <div class="metric-value">{n_dup:,}</div>
            <div class="metric-label">Duplicados</div>
        </div>""", unsafe_allow_html=True)
    with c8:
        completeness = round((1 - null_total / (meta['n_rows'] * meta['n_cols'])) * 100, 1)
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:1.5rem">✅</div>
            <div class="metric-value">{completeness}%</div>
            <div class="metric-label">Completitud</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Vista previa del dataset
    st.markdown("### 👁️ Vista Previa del Dataset")
    n_preview = st.slider("Filas a mostrar", 5, min(100, len(df)), 10, key='eda_preview_rows')
    
    # Aplicar estilos a la tabla
    st.dataframe(
        df.head(n_preview),
        use_container_width=True,
        hide_index=False,
    )
    
    # Tipos de datos
    st.markdown("### 🏷️ Tipos de Variables Detectados")
    col_a, col_b = st.columns(2)
    
    with col_a:
        if meta['numeric_cols']:
            st.markdown("**🔢 Variables Numéricas:**")
            for c in meta['numeric_cols']:
                st.markdown(f'<span class="badge badge-primary">📊 {c}</span>', unsafe_allow_html=True)
        
        if meta['datetime_cols']:
            st.markdown("<br>**📅 Variables de Fecha:**")
            for c in meta['datetime_cols']:
                st.markdown(f'<span class="badge badge-info">📅 {c}</span>', unsafe_allow_html=True)
    
    with col_b:
        if meta['categorical_cols']:
            st.markdown("**🏷️ Variables Categóricas:**")
            for c in meta['categorical_cols']:
                st.markdown(f'<span class="badge badge-warning">🏷️ {c}</span>', unsafe_allow_html=True)
        
        if meta['id_like_cols']:
            st.markdown("<br>**🆔 Posibles IDs:**")
            for c in meta['id_like_cols']:
                st.markdown(f'<span class="badge badge-danger">🆔 {c}</span>', unsafe_allow_html=True)
    
    # Insights automáticos
    st.markdown("### 💡 Insights Automáticos")
    insights = _generate_insights(df, meta)
    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)


def _generate_insights(df: pd.DataFrame, meta: Dict) -> List[str]:
    """Genera insights automáticos sobre el dataset."""
    insights = []
    n_rows, n_cols = meta['n_rows'], meta['n_cols']
    
    # Tamaño
    if n_rows < 100:
        insights.append("⚠️ <b>Dataset pequeño</b>: Menos de 100 registros. Los modelos pueden tener alta varianza y los resultados pueden no ser representativos.")
    elif n_rows > 100000:
        insights.append(f"🚀 <b>Dataset grande</b>: {n_rows:,} registros. Considera usar muestras para exploraciones rápidas.")
    else:
        insights.append(f"✅ <b>Tamaño adecuado</b>: {n_rows:,} filas y {n_cols} columnas. Dataset de tamaño razonable para ML.")
    
    # Nulos
    null_total = sum(meta['null_counts'].values())
    null_pct_total = null_total / (n_rows * n_cols) * 100
    if null_pct_total == 0:
        insights.append("✅ <b>Sin valores nulos</b>: El dataset está completo. No se requiere imputación.")
    elif null_pct_total < 5:
        insights.append(f"⚠️ <b>Pocos valores nulos</b>: {null_pct_total:.1f}% de nulos. Se aplicará imputación simple.")
    elif null_pct_total < 30:
        insights.append(f"⚠️ <b>Valores nulos moderados</b>: {null_pct_total:.1f}% de nulos. Se recomienda análisis cuidadoso antes de imputar.")
    else:
        insights.append(f"❌ <b>Alto porcentaje de nulos</b>: {null_pct_total:.1f}% de nulos. Considera eliminar columnas con >50% de nulos.")
    
    # Duplicados
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        insights.append(f"⚠️ <b>Filas duplicadas</b>: {n_dup} filas duplicadas ({n_dup/n_rows*100:.1f}%). Considera eliminarlas en el preprocesamiento.")
    
    # Balance categórico para targets potenciales
    for target in meta['potential_targets'][:2]:
        vc = df[target].value_counts(normalize=True)
        if len(vc) >= 2:
            max_class = vc.max()
            if max_class > 0.8:
                insights.append(f"⚠️ <b>Desbalance detectado en '{target}'</b>: La clase dominante tiene {max_class*100:.1f}% de los datos. Considera técnicas de balanceo (SMOTE, class_weight).")
    
    # Correlaciones altas
    num_cols = meta['numeric_cols']
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        high_corr = [(corr.index[i], corr.columns[j], corr.iloc[i,j])
                     for i in range(len(corr)) for j in range(i+1, len(corr))
                     if corr.iloc[i,j] > 0.9]
        if high_corr:
            pairs = ', '.join([f"'{a}' y '{b}' ({v:.2f})" for a,b,v in high_corr[:3]])
            insights.append(f"🔗 <b>Alta correlación detectada</b>: {pairs}. Puede existir multicolinealidad.")
    
    return insights


# ─── Tab: Estadísticas ────────────────────────────────────────────────────────
def _tab_estadisticas(df: pd.DataFrame, meta: Dict) -> None:
    st.markdown("### 📊 Estadísticas Descriptivas")
    
    col_filter = st.selectbox(
        "Tipo de variables:", 
        ["Todas (numéricas)", "Por columna específica"],
        key='eda_stats_filter'
    )
    
    if col_filter == "Todas (numéricas)":
        if meta['numeric_cols']:
            desc = df[meta['numeric_cols']].describe().T
            desc.index.name = 'Variable'
            st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)
        else:
            st.info("No hay variables numéricas en el dataset.")
    else:
        col = st.selectbox("Selecciona columna:", df.columns.tolist(), key='eda_col_detail')
        if col:
            summary = get_column_summary(df, col)
            _render_column_detail(summary, df, col, meta)
    
    # Tabla de tipos y nulos
    st.markdown("### 📋 Resumen por Columna")
    summary_data = []
    for c in df.columns:
        null_c = meta['null_counts'].get(c, 0)
        null_p = meta['null_pct'].get(c, 0)
        tipo = ('🔢 Numérica' if c in meta['numeric_cols'] 
                else '🏷️ Categórica' if c in meta['categorical_cols']
                else '📅 Fecha' if c in meta['datetime_cols']
                else '📝 Texto')
        summary_data.append({
            'Columna': c,
            'Tipo': tipo,
            'Dtype': meta['dtypes'].get(c, ''),
            'Nulos': null_c,
            'Nulos %': f"{null_p:.1f}%",
            'Únicos': df[c].nunique(),
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


def _render_column_detail(summary: Dict, df: pd.DataFrame, col: str, meta: Dict) -> None:
    """Detalle visual de una columna específica."""
    c1, c2 = st.columns([1, 2])
    
    with c1:
        is_num = col in meta['numeric_cols']
        items = [
            ("Tipo", summary['dtype']),
            ("Total", f"{summary['n_total']:,}"),
            ("Válidos", f"{summary['n_valid']:,}"),
            ("Nulos", f"{summary['n_null']} ({summary['null_pct']}%)"),
            ("Únicos", f"{summary['n_unique']} ({summary['unique_pct']}%)"),
        ]
        if is_num:
            items += [
                ("Media", f"{summary.get('mean', 'N/A')}"),
                ("Desv. Estándar", f"{summary.get('std', 'N/A')}"),
                ("Mediana", f"{summary.get('median', 'N/A')}"),
                ("Asimetría", f"{summary.get('skew', 'N/A')}"),
                ("Curtosis", f"{summary.get('kurt', 'N/A')}"),
                ("Mínimo", f"{summary.get('min', 'N/A')}"),
                ("Máximo", f"{summary.get('max', 'N/A')}"),
            ]
        else:
            items += [("Moda", str(summary.get('mode', 'N/A')))]
        
        for label, val in items:
            st.markdown(f"**{label}:** `{val}`")
    
    with c2:
        if col in meta['numeric_cols']:
            st.plotly_chart(histogram(df, col), use_container_width=True)
        else:
            st.plotly_chart(bar_categorical(df, col), use_container_width=True)


# ─── Tab: Distribuciones ──────────────────────────────────────────────────────
def _tab_distribuciones(df: pd.DataFrame, meta: Dict) -> None:
    st.markdown("### 📈 Distribución de Variables")
    
    subtab1, subtab2, subtab3 = st.tabs(["Numéricas", "Categóricas", "Pairplot"])
    
    with subtab1:
        num_cols = meta['numeric_cols']
        if not num_cols:
            st.info("No hay variables numéricas.")
            return
        
        col_sel = st.multiselect(
            "Variables a visualizar:", num_cols,
            default=num_cols[:min(4, len(num_cols))],
            key='eda_dist_num'
        )
        chart_type = st.radio("Tipo de gráfico:", ["Histograma", "Boxplot"], horizontal=True, key='eda_chart_type')
        
        if col_sel:
            cols_per_row = 2
            rows = [col_sel[i:i+cols_per_row] for i in range(0, len(col_sel), cols_per_row)]
            for row in rows:
                cols = st.columns(len(row))
                for idx, c in enumerate(row):
                    with cols[idx]:
                        if chart_type == "Histograma":
                            st.plotly_chart(histogram(df, c), use_container_width=True)
                        else:
                            st.plotly_chart(boxplot(df, c), use_container_width=True)
    
    with subtab2:
        cat_cols = meta['categorical_cols']
        if not cat_cols:
            st.info("No hay variables categóricas.")
            return
        
        col_sel_cat = st.multiselect(
            "Variables categóricas:", cat_cols,
            default=cat_cols[:min(4, len(cat_cols))],
            key='eda_dist_cat'
        )
        
        if col_sel_cat:
            cols_per_row = 2
            rows = [col_sel_cat[i:i+cols_per_row] for i in range(0, len(col_sel_cat), cols_per_row)]
            for row in rows:
                cols = st.columns(len(row))
                for idx, c in enumerate(row):
                    with cols[idx]:
                        st.plotly_chart(bar_categorical(df, c), use_container_width=True)
    
    with subtab3:
        num_cols = meta['numeric_cols']
        if len(num_cols) < 2:
            st.info("Se necesitan al menos 2 variables numéricas para el pairplot.")
            return
        
        pair_cols = st.multiselect(
            "Variables para pairplot:", num_cols,
            default=num_cols[:min(5, len(num_cols))],
            key='eda_pairplot_cols'
        )
        color_col = st.selectbox(
            "Color por:", ['(Ninguno)'] + meta['categorical_cols'][:10],
            key='eda_pairplot_color'
        )
        
        if len(pair_cols) >= 2:
            color = color_col if color_col != '(Ninguno)' else None
            with st.spinner("Generando pairplot..."):
                st.plotly_chart(
                    pairplot_sample(df, pair_cols, color_col=color),
                    use_container_width=True
                )


# ─── Tab: Correlaciones ───────────────────────────────────────────────────────
def _tab_correlaciones(df: pd.DataFrame, meta: Dict) -> None:
    st.markdown("### 🔗 Análisis de Correlaciones")
    
    num_cols = meta['numeric_cols']
    if len(num_cols) < 2:
        st.info("Se necesitan al menos 2 variables numéricas para calcular correlaciones.")
        return
    
    cols_sel = st.multiselect(
        "Variables a incluir:", num_cols,
        default=num_cols[:min(12, len(num_cols))],
        key='eda_corr_cols'
    )
    
    if len(cols_sel) >= 2:
        with st.spinner("Calculando correlaciones..."):
            st.plotly_chart(correlation_heatmap(df, cols_sel), use_container_width=True)
        
        # Top correlaciones
        corr_matrix = df[cols_sel].corr()
        corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_pairs.append({
                    'Variable A': corr_matrix.index[i],
                    'Variable B': corr_matrix.columns[j],
                    'Correlación': round(corr_matrix.iloc[i,j], 4),
                    '|Correlación|': round(abs(corr_matrix.iloc[i,j]), 4),
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values('|Correlación|', ascending=False)
        
        st.markdown("#### 🏆 Top Correlaciones")
        n_top = st.slider("Mostrar top:", 5, min(30, len(corr_df)), 10, key='eda_top_corr')
        
        top_corr = corr_df.head(n_top)[['Variable A', 'Variable B', 'Correlación']].copy()
        
        def color_corr(val):
            color = 'rgba(16,185,129,0.3)' if val > 0 else 'rgba(239,68,68,0.3)'
            intensity = abs(val)
            return f'background-color: {color}; font-weight: {"bold" if intensity > 0.7 else "normal"}'
        
        st.dataframe(
            top_corr.style.applymap(color_corr, subset=['Correlación']),
            use_container_width=True,
            hide_index=True,
        )
        
        # Scatter de las top 2 correlaciones
        if len(corr_df) >= 1:
            top1 = corr_df.iloc[0]
            st.markdown(f"#### 🔍 Scatter: {top1['Variable A']} vs {top1['Variable B']}")
            color_by = st.selectbox(
                "Color por:", ['(Ninguno)'] + meta['categorical_cols'][:8],
                key='eda_scatter_color'
            )
            color_val = color_by if color_by != '(Ninguno)' else None
            st.plotly_chart(
                scatter_2d(df, top1['Variable A'], top1['Variable B'], color=color_val),
                use_container_width=True
            )


# ─── Tab: Valores Nulos ───────────────────────────────────────────────────────
def _tab_nulos(df: pd.DataFrame, meta: Dict) -> None:
    st.markdown("### 🚫 Análisis de Valores Nulos")
    
    null_counts = {k: v for k, v in meta['null_counts'].items() if v > 0}
    
    if not null_counts:
        st.markdown("""
        <div class="insight-box">
            ✅ <b>¡Dataset completo!</b> No se encontraron valores nulos en ninguna columna.
            Esto es ideal para el análisis y entrenamiento de modelos.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Barra de nulos
    null_df = pd.DataFrame({
        'Columna': list(null_counts.keys()),
        'Nulos': list(null_counts.values()),
        'Porcentaje': [meta['null_pct'][k] for k in null_counts.keys()],
    }).sort_values('Nulos', ascending=False)
    
    fig = go.Figure(go.Bar(
        x=null_df['Porcentaje'],
        y=null_df['Columna'],
        orientation='h',
        marker=dict(
            color=null_df['Porcentaje'],
            colorscale=[[0, '#10B981'], [0.3, '#F59E0B'], [0.7, '#EF4444'], [1, '#991B1B']],
            showscale=True,
            colorbar=dict(title='% Nulos'),
        ),
        text=[f'{p:.1f}%' for p in null_df['Porcentaje']],
        textposition='outside',
        hovertemplate='%{y}<br>Nulos: %{x:.1f}%<extra></extra>',
    ))
    apply_layout(fig, 'Porcentaje de Valores Nulos por Columna', height=max(350, len(null_df)*30+80))
    fig.update_xaxes(title_text='Porcentaje (%)', range=[0, 115])
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap de nulos
    st.markdown("#### 🗺️ Mapa de Nulos")
    cols_with_nulls = [c for c, v in null_counts.items() if v > 0]
    if cols_with_nulls:
        st.plotly_chart(null_heatmap(df[cols_with_nulls]), use_container_width=True)
    
    # Tabla detallada
    st.dataframe(null_df, use_container_width=True, hide_index=True)
    
    # Recomendaciones
    st.markdown("#### 💡 Recomendaciones")
    for _, row in null_df.iterrows():
        if row['Porcentaje'] > 50:
            msg = f"❌ <b>'{row['Columna']}'</b>: {row['Porcentaje']:.1f}% nulos → Considera <b>eliminar esta columna</b>"
            badge = 'danger'
        elif row['Porcentaje'] > 20:
            msg = f"⚠️ <b>'{row['Columna']}'</b>: {row['Porcentaje']:.1f}% nulos → Imputación con <b>mediana/moda</b> o modelo de imputación"
            badge = 'warning'
        else:
            msg = f"ℹ️ <b>'{row['Columna']}'</b>: {row['Porcentaje']:.1f}% nulos → Imputación simple con <b>mediana/moda</b>"
            badge = 'info'
        st.markdown(f'<div class="insight-box">{msg}</div>', unsafe_allow_html=True)


# ─── Tab: Outliers ────────────────────────────────────────────────────────────
def _tab_outliers(df: pd.DataFrame, meta: Dict) -> None:
    st.markdown("### ⚠️ Detección de Outliers (Método IQR)")
    
    num_cols = meta['numeric_cols']
    if not num_cols:
        st.info("No hay variables numéricas para analizar outliers.")
        return
    
    outlier_results = []
    for col in num_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((col_data < lower) | (col_data > upper)).sum()
        pct = n_outliers / len(col_data) * 100
        outlier_results.append({
            'Variable': col,
            'N Outliers': n_outliers,
            '% Outliers': round(pct, 2),
            'Límite Inferior': round(lower, 4),
            'Límite Superior': round(upper, 4),
            'Min Real': round(col_data.min(), 4),
            'Max Real': round(col_data.max(), 4),
        })
    
    if not outlier_results:
        st.info("No se pudieron calcular outliers (IQR = 0 en todas las columnas).")
        return
    
    out_df = pd.DataFrame(outlier_results).sort_values('% Outliers', ascending=False)
    
    # Gráfico de outliers
    fig = go.Figure(go.Bar(
        x=out_df['Variable'],
        y=out_df['% Outliers'],
        marker=dict(
            color=out_df['% Outliers'],
            colorscale=[[0, '#10B981'], [0.2, '#F59E0B'], [0.5, '#EF4444'], [1, '#991B1B']],
            showscale=False,
        ),
        text=[f"{v:.1f}%" for v in out_df['% Outliers']],
        textposition='outside',
        hovertemplate='%{x}<br>Outliers: %{y:.2f}%<extra></extra>',
    ))
    apply_layout(fig, 'Porcentaje de Outliers por Variable', height=380)
    st.plotly_chart(fig, use_container_width=True)
    
    # Boxplots interactivos
    col_sel = st.selectbox("Ver boxplot de:", num_cols, key='eda_outlier_box')
    if col_sel:
        st.plotly_chart(boxplot(df, col_sel), use_container_width=True)
    
    # Tabla resumen
    st.dataframe(out_df, use_container_width=True, hide_index=True)


# ─── Tab: Análisis de Clases ─────────────────────────────────────────────────
def _tab_clases(df: pd.DataFrame, meta: Dict) -> None:
    st.markdown("### 🎯 Análisis de Variable Objetivo")
    
    if not meta['potential_targets']:
        st.info("No se detectaron columnas objetivo potenciales (variables con 2-20 clases únicas).")
        return
    
    target = st.selectbox(
        "Selecciona variable objetivo para analizar:",
        meta['potential_targets'],
        key='eda_target_col'
    )
    
    if target:
        vc = df[target].value_counts()
        vc_pct = df[target].value_counts(normalize=True) * 100
        
        c1, c2 = st.columns([1, 1])
        with c1:
            # Donut de clases
            fig = go.Figure(go.Pie(
                labels=vc.index.astype(str),
                values=vc.values,
                hole=0.45,
                marker_colors=PALETTE_PRIMARY[:len(vc)],
                textinfo='label+percent',
                hovertemplate='%{label}<br>%{value} registros (%{percent})<extra></extra>',
            ))
            fig.update_layout(
                title_text=f'Distribución — {target}',
                height=380,
                paper_bgcolor='rgba(15,15,26,0)',
                font=dict(color='#E2E8F0'),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            # Barras con conteos
            fig2 = go.Figure(go.Bar(
                x=vc.index.astype(str),
                y=vc.values,
                marker_color=PALETTE_PRIMARY[:len(vc)],
                text=[f'{v} ({p:.1f}%)' for v, p in zip(vc.values, vc_pct.values)],
                textposition='outside',
                hovertemplate='%{x}<br>Conteo: %{y}<extra></extra>',
            ))
            apply_layout(fig2, f'Conteo por Clase — {target}', height=380)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Tabla de clases
        class_df = pd.DataFrame({
            'Clase': vc.index.astype(str),
            'Conteo': vc.values,
            'Porcentaje': [f"{p:.2f}%" for p in vc_pct.values],
        })
        st.dataframe(class_df, use_container_width=True, hide_index=True)
        
        # Balance assessment
        max_pct = vc_pct.max()
        n_classes = len(vc)
        
        if n_classes == 2:
            if max_pct > 80:
                msg = f"❌ <b>Dataset muy desbalanceado</b>: La clase dominante tiene {max_pct:.1f}%. Se recomienda usar class_weight='balanced', SMOTE o técnicas de oversampling."
            elif max_pct > 60:
                msg = f"⚠️ <b>Leve desbalance</b>: {max_pct:.1f}% en la clase dominante. Monitorea el recall de la clase minoritaria."
            else:
                msg = f"✅ <b>Dataset balanceado</b>: Distribución relativamente uniforme entre las {n_classes} clases."
        else:
            if max_pct > 70:
                msg = f"❌ <b>Dataset muy desbalanceado</b> ({n_classes} clases): La clase dominante tiene {max_pct:.1f}%. Se recomienda estratificación y métricas ponderadas."
            else:
                msg = f"✅ <b>Balance aceptable</b> ({n_classes} clases): Distribución razonable para clasificación multiclase."
        
        st.markdown(f'<div class="insight-box">{msg}</div>', unsafe_allow_html=True)
        
        # Cross-analysis con numéricas
        num_cols = meta['numeric_cols']
        if num_cols:
            st.markdown("#### 📊 Distribución por Clase")
            num_sel = st.selectbox("Variable numérica:", num_cols, key='eda_class_num')
            if num_sel:
                st.plotly_chart(boxplot(df, num_sel, group_by=target), use_container_width=True)
