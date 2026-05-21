"""
visualization.py
================
Utilidades de visualización compartidas entre módulos.
Usa Plotly para gráficos interactivos.
"""
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional


# ─── Paleta Corporativa ────────────────────────────────────────────────────────
PALETTE_PRIMARY   = ['#7C3AED', '#A78BFA', '#06B6D4', '#10B981', '#F59E0B',
                     '#EF4444', '#8B5CF6', '#67E8F9', '#6EE7B7', '#FCD34D']
PALETTE_GRADIENT  = ['#7C3AED', '#6D28D9', '#5B21B6', '#4C1D95']
PALETTE_DIVERGING = ['#EF4444', '#F97316', '#EAB308', '#22C55E', '#06B6D4', '#7C3AED']

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(15,15,26,0)',
    plot_bgcolor='rgba(26,26,46,0.5)',
    font=dict(family='Inter, sans-serif', color='#E2E8F0', size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor='rgba(26,26,46,0.8)',
        bordercolor='rgba(124,58,237,0.3)',
        borderwidth=1,
    ),
    xaxis=dict(
        gridcolor='rgba(148,163,184,0.1)',
        linecolor='rgba(148,163,184,0.2)',
        zerolinecolor='rgba(148,163,184,0.15)',
    ),
    yaxis=dict(
        gridcolor='rgba(148,163,184,0.1)',
        linecolor='rgba(148,163,184,0.2)',
        zerolinecolor='rgba(148,163,184,0.15)',
    ),
)


def apply_layout(fig: go.Figure, title: str = '', height: int = 400) -> go.Figure:
    """Aplica el tema corporativo a una figura Plotly."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color='#A78BFA', family='Inter')),
        height=height,
        **PLOTLY_LAYOUT
    )
    import streamlit as st
    if 'pdf_figures' not in st.session_state:
        st.session_state['pdf_figures'] = {}
    if title:
        st.session_state['pdf_figures'][title] = fig
    return fig


# ─── Histogramas ──────────────────────────────────────────────────────────────
def histogram(df: pd.DataFrame, col: str, bins: int = 30, color: str = None) -> go.Figure:
    """Histograma con KDE overlay para una variable numérica."""
    c = color or PALETTE_PRIMARY[0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[col].dropna(),
        nbinsx=bins,
        marker_color=c,
        opacity=0.75,
        name=col,
        hovertemplate='Valor: %{x}<br>Frecuencia: %{y}<extra></extra>',
    ))
    apply_layout(fig, f'Distribución — {col}', height=380)
    fig.update_xaxes(title_text=col)
    fig.update_yaxes(title_text='Frecuencia')
    return fig


# ─── Boxplots ─────────────────────────────────────────────────────────────────
def boxplot(df: pd.DataFrame, col: str, group_by: str = None) -> go.Figure:
    """Boxplot para una variable, opcionalmente agrupado."""
    fig = go.Figure()
    if group_by and group_by in df.columns:
        groups = df[group_by].dropna().unique()
        for i, grp in enumerate(groups):
            subset = df[df[group_by] == grp][col].dropna()
            fig.add_trace(go.Box(
                y=subset, name=str(grp),
                marker_color=PALETTE_PRIMARY[i % len(PALETTE_PRIMARY)],
                boxmean=True,
            ))
    else:
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col,
            marker_color=PALETTE_PRIMARY[0],
            boxmean=True,
            fillcolor='rgba(124,58,237,0.15)',
        ))
    apply_layout(fig, f'Boxplot — {col}', height=380)
    return fig


# ─── Barras para Categóricas ──────────────────────────────────────────────────
def bar_categorical(df: pd.DataFrame, col: str, max_cats: int = 20) -> go.Figure:
    """Gráfico de barras para variable categórica."""
    vc = df[col].value_counts().head(max_cats)
    fig = go.Figure(go.Bar(
        x=vc.index.astype(str),
        y=vc.values,
        marker=dict(
            color=vc.values,
            colorscale=[[0, '#4C1D95'], [1, '#7C3AED']],
            showscale=False,
        ),
        text=vc.values,
        textposition='outside',
        hovertemplate='%{x}<br>Conteo: %{y}<extra></extra>',
    ))
    apply_layout(fig, f'Distribución — {col}', height=380)
    fig.update_xaxes(tickangle=-35)
    return fig


# ─── Heatmap de Correlación ───────────────────────────────────────────────────
def correlation_heatmap(df: pd.DataFrame, cols: List[str]) -> go.Figure:
    """Heatmap de matriz de correlación Pearson."""
    corr = df[cols].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # solo triángulo inferior
    
    z = corr.values.copy()
    text = np.round(z, 2).astype(str)
    
    fig = go.Figure(go.Heatmap(
        z=z,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        text=text,
        texttemplate='%{text}',
        colorscale='RdBu_r',
        zmid=0, zmin=-1, zmax=1,
        hovertemplate='%{y} vs %{x}<br>Correlación: %{z:.3f}<extra></extra>',
        colorbar=dict(title='Correlación', tickfont=dict(color='#E2E8F0')),
    ))
    n = len(cols)
    height = max(400, min(700, n * 40 + 100))
    apply_layout(fig, 'Matriz de Correlación', height=height)
    fig.update_xaxes(tickangle=-40)
    return fig


# ─── Scatter Plot ─────────────────────────────────────────────────────────────
def scatter_2d(df: pd.DataFrame, x: str, y: str,
               color: str = None, size: str = None,
               labels_map: dict = None) -> go.Figure:
    """Scatterplot 2D interactivo."""
    kwargs = dict(x=x, y=y, hover_data=df.columns.tolist()[:6])
    if color and color in df.columns:
        kwargs['color'] = color
    if size and size in df.columns:
        kwargs['size'] = size
    if labels_map:
        kwargs['labels'] = labels_map
    
    fig = px.scatter(df, **kwargs, color_discrete_sequence=PALETTE_PRIMARY)
    apply_layout(fig, f'{x} vs {y}', height=420)
    return fig


# ─── Null Heatmap ─────────────────────────────────────────────────────────────
def null_heatmap(df: pd.DataFrame, max_rows: int = 200) -> go.Figure:
    """Heatmap de valores faltantes."""
    sample = df.head(max_rows) if len(df) > max_rows else df
    z = sample.isnull().astype(int).T.values
    
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f'Fila {i}' for i in range(len(sample))],
        y=sample.columns.tolist(),
        colorscale=[[0, 'rgba(16,185,129,0.3)'], [1, 'rgba(239,68,68,0.8)']],
        showscale=False,
        hovertemplate='%{y}<br>Fila %{x}<br>%{z}<extra></extra>',
    ))
    height = max(300, len(df.columns) * 22 + 60)
    apply_layout(fig, f'Mapa de Valores Nulos (primeras {len(sample)} filas)', height=height)
    return fig


# ─── Feature Importance ───────────────────────────────────────────────────────
def feature_importance_chart(feature_names: List[str],
                              importances: np.ndarray,
                              title: str = 'Importancia de Variables') -> go.Figure:
    """Gráfico de importancia de variables (horizontal)."""
    sorted_idx = np.argsort(importances)
    names_sorted = [str(feature_names[i]) for i in sorted_idx]
    vals_sorted  = importances[sorted_idx]
    
    colors = [PALETTE_PRIMARY[0] if v >= np.median(vals_sorted)
              else PALETTE_PRIMARY[2] for v in vals_sorted]
    
    fig = go.Figure(go.Bar(
        x=vals_sorted,
        y=names_sorted,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in vals_sorted],
        textposition='outside',
        hovertemplate='%{y}<br>Importancia: %{x:.4f}<extra></extra>',
    ))
    height = max(400, len(feature_names) * 25 + 80)
    apply_layout(fig, title, height=height)
    fig.update_xaxes(title_text='Importancia')
    return fig


# ─── ROC Curve ────────────────────────────────────────────────────────────────
def roc_curve_plot(fpr_dict: dict, tpr_dict: dict, auc_dict: dict) -> go.Figure:
    """
    Curvas ROC para uno o múltiples modelos/clases.
    fpr_dict, tpr_dict, auc_dict: {nombre_modelo: array}
    """
    fig = go.Figure()
    
    # Línea base aleatoria
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random (AUC=0.5)',
        line=dict(color='rgba(148,163,184,0.4)', width=1, dash='dash'),
    ))
    
    for i, (name, fpr) in enumerate(fpr_dict.items()):
        tpr = tpr_dict[name]
        auc = auc_dict.get(name, 0)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{name} (AUC={auc:.3f})',
            line=dict(color=PALETTE_PRIMARY[i % len(PALETTE_PRIMARY)], width=2.5),
            fill='tozeroy' if i == 0 else None,
            fillcolor='rgba(124,58,237,0.07)' if i == 0 else None,
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>',
        ))
    
    apply_layout(fig, 'Curva ROC', height=450)
    fig.update_xaxes(title_text='Tasa de Falsos Positivos (FPR)', range=[-0.02, 1.02])
    fig.update_yaxes(title_text='Tasa de Verdaderos Positivos (TPR)', range=[-0.02, 1.02])
    return fig


# ─── Confusion Matrix ─────────────────────────────────────────────────────────
def confusion_matrix_plot(cm: np.ndarray, class_labels: List[str]) -> go.Figure:
    """Heatmap interactivo de matriz de confusión."""
    pct = cm / cm.sum() * 100
    text = [[f'{cm[i,j]}<br>({pct[i,j]:.1f}%)' for j in range(cm.shape[1])]
            for i in range(cm.shape[0])]
    
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[str(l) for l in class_labels],
        y=[str(l) for l in class_labels],
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=13, color='white', family='Inter'),
        colorscale=[[0, '#1A1A2E'], [0.5, '#5B21B6'], [1, '#7C3AED']],
        hovertemplate='Real: %{y}<br>Predicho: %{x}<br>Conteo: %{z}<extra></extra>',
        showscale=True,
        colorbar=dict(title='Conteo', tickfont=dict(color='#E2E8F0')),
    ))
    apply_layout(fig, 'Matriz de Confusión', height=420)
    fig.update_xaxes(title_text='Predicho', side='bottom')
    fig.update_yaxes(title_text='Real', autorange='reversed')
    return fig


# ─── Cluster 2D ───────────────────────────────────────────────────────────────
def cluster_scatter(components: np.ndarray, labels: np.ndarray,
                    method: str = 'PCA', centroids: np.ndarray = None) -> go.Figure:
    """Scatter 2D de clusters con colores diferenciados."""
    unique_labels = np.unique(labels)
    fig = go.Figure()
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        fig.add_trace(go.Scatter(
            x=components[mask, 0],
            y=components[mask, 1],
            mode='markers',
            name=f'Cluster {label}',
            marker=dict(
                color=PALETTE_PRIMARY[i % len(PALETTE_PRIMARY)],
                size=5,
                opacity=0.75,
                line=dict(width=0.3, color='rgba(0,0,0,0.3)'),
            ),
            hovertemplate=f'Cluster {label}<br>{method}1: %{{x:.3f}}<br>{method}2: %{{y:.3f}}<extra></extra>',
        ))
    
    if centroids is not None:
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1],
            mode='markers',
            name='Centroides',
            marker=dict(symbol='x', size=14, color='white',
                        line=dict(width=2, color='white')),
        ))
    
    apply_layout(fig, f'Visualización de Clusters ({method})', height=480)
    fig.update_xaxes(title_text=f'{method} Componente 1')
    fig.update_yaxes(title_text=f'{method} Componente 2')
    return fig


# ─── Elbow Chart ──────────────────────────────────────────────────────────────
def elbow_chart(k_range: List[int], inertias: List[float],
                silhouettes: List[float] = None, best_k: int = None) -> go.Figure:
    """Método del codo con silhouette overlay."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(
        x=k_range, y=inertias,
        mode='lines+markers',
        name='Inercia',
        line=dict(color=PALETTE_PRIMARY[0], width=2.5),
        marker=dict(size=8),
    ), secondary_y=False)
    
    if silhouettes:
        fig.add_trace(go.Scatter(
            x=k_range, y=silhouettes,
            mode='lines+markers',
            name='Silhouette',
            line=dict(color=PALETTE_PRIMARY[2], width=2.5, dash='dot'),
            marker=dict(size=8),
        ), secondary_y=True)
    
    if best_k:
        fig.add_vline(
            x=best_k, line_dash='dash',
            line_color='rgba(16,185,129,0.8)',
            annotation_text=f'K={best_k}',
            annotation_font_color='#6EE7B7',
        )
    
    apply_layout(fig, 'Método del Codo — Selección de K', height=400)
    fig.update_xaxes(title_text='Número de Clusters (K)', dtick=1)
    fig.update_yaxes(title_text='Inercia (WCSS)', secondary_y=False)
    if silhouettes:
        fig.update_yaxes(title_text='Silhouette Score', secondary_y=True,
                         showgrid=False)
    return fig


# ─── Split Visualization ──────────────────────────────────────────────────────
def split_donut(train_pct: float, val_pct: float, test_pct: float,
                n_total: int) -> go.Figure:
    """Donut chart de la partición train/val/test."""
    labels = ['Entrenamiento', 'Validación', 'Prueba']
    values = [
        int(n_total * train_pct / 100),
        int(n_total * val_pct / 100),
        int(n_total * test_pct / 100),
    ]
    colors = [PALETTE_PRIMARY[0], PALETTE_PRIMARY[2], PALETTE_PRIMARY[4]]
    
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker_colors=colors,
        textinfo='label+percent',
        textfont=dict(size=13, color='white'),
        hovertemplate='%{label}<br>%{value} muestras (%{percent})<extra></extra>',
    ))
    
    fig.update_layout(
        annotations=[dict(
            text=f'<b>{n_total}</b><br>total',
            x=0.5, y=0.5, font_size=16,
            font_color='#E2E8F0', showarrow=False,
        )],
    )
    apply_layout(fig, 'Partición de Datos Train/Val/Test', height=360)
    return fig


# ─── Metric Comparison Bar ────────────────────────────────────────────────────
def model_comparison_bar(results_df: pd.DataFrame, metric: str = 'F1-Score') -> go.Figure:
    """Barras comparativas de modelos."""
    df_sorted = results_df.sort_values(metric, ascending=False)
    
    fig = go.Figure(go.Bar(
        x=df_sorted['Modelo'],
        y=df_sorted[metric],
        marker=dict(
            color=df_sorted[metric],
            colorscale=[[0, '#4C1D95'], [0.5, '#7C3AED'], [1, '#06B6D4']],
            showscale=False,
        ),
        text=[f'{v:.3f}' for v in df_sorted[metric]],
        textposition='outside',
        hovertemplate='%{x}<br>' + metric + ': %{y:.4f}<extra></extra>',
    ))
    
    apply_layout(fig, f'Comparación de Modelos — {metric}', height=380)
    fig.update_yaxes(title_text=metric, range=[0, min(1.15, df_sorted[metric].max() * 1.15)])
    return fig


# ─── Pairplot (Sample) ────────────────────────────────────────────────────────
def pairplot_sample(df: pd.DataFrame, cols: List[str],
                    color_col: str = None, max_n: int = 500) -> go.Figure:
    """Pairplot simplificado usando Plotly Splom."""
    sample = df.sample(min(max_n, len(df)), random_state=42)
    
    dims = [dict(label=c, values=sample[c]) for c in cols if c in sample.columns]
    
    kwargs = dict(dimensions=dims,
                  marker=dict(size=3, opacity=0.6,
                              color=PALETTE_PRIMARY[0]),
                  diagonal_visible=True)
    
    if color_col and color_col in sample.columns:
        unique_vals = sample[color_col].astype('category').cat.codes
        kwargs['marker']['color'] = unique_vals
        kwargs['marker']['colorscale'] = PALETTE_PRIMARY[:len(sample[color_col].unique())]
    
    fig = go.Figure(go.Splom(**kwargs))
    apply_layout(fig, 'Pairplot — Variables Numéricas', height=600)
    return fig
