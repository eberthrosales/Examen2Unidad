"""
clustering.py
=============
Segmentación no supervisada: K-Means y Clustering Jerárquico.
Detección automática de variables, normalización y visualización.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from modules.visualization import (
    cluster_scatter, elbow_chart, apply_layout, PALETTE_PRIMARY
)
from modules.interpretation import interpret_clusters


# ─── Sección Principal ────────────────────────────────────────────────────────
def render_clustering(df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔮</div>
        <div>
            <div class="section-title">Segmentación (Clustering)</div>
            <div class="section-subtitle">Agrupamiento no supervisado: K-Means y Jerárquico</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs([
        "⚙️ Configuración", "📊 Selección de K", "🗂️ K-Means",
        "🌳 Jerárquico", "🗺️ Visualización", "📋 Perfiles"
    ])
    
    with tabs[0]: _tab_config_cluster(df, meta)
    with tabs[1]: _tab_elbow(df, meta)
    with tabs[2]: _tab_kmeans(df, meta)
    with tabs[3]: _tab_hierarchical(df, meta)
    with tabs[4]: _tab_viz_cluster(df, meta)
    with tabs[5]: _tab_profiles(df, meta)


# ─── Preparar datos para clustering ──────────────────────────────────────────
def _prepare_cluster_data(df: pd.DataFrame, meta: Dict,
                           selected_cols: List[str]) -> Optional[np.ndarray]:
    """Prepara y normaliza datos para clustering."""
    try:
        X = df[selected_cols].copy()
        
        # Imputar nulos
        for col in selected_cols:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Escalar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        st.session_state['cluster_scaler'] = scaler
        st.session_state['cluster_cols'] = selected_cols
        
        return X_scaled
    
    except Exception as e:
        st.error(f"❌ Error preparando datos: {str(e)}")
        return None


# ─── Tab: Configuración ───────────────────────────────────────────────────────
def _tab_config_cluster(df: pd.DataFrame, meta: Dict) -> None:
    
    num_cols = meta.get('numeric_cols', [])
    
    if not num_cols:
        st.error("❌ No hay variables numéricas para realizar clustering.")
        return
    
    st.markdown("#### 📐 Variables para Clustering")
    
    # Filtrar por nulos extremos
    valid_cols = [c for c in num_cols if meta['null_pct'].get(c, 0) < 50]
    
    selected_cols = st.multiselect(
        "Selecciona variables numéricas:",
        valid_cols,
        default=valid_cols[:min(8, len(valid_cols))],
        key='cluster_cols_sel',
        help="Se recomienda usar entre 2 y 15 variables. Se normalizarán automáticamente."
    )
    
    if not selected_cols:
        st.warning("⚠️ Selecciona al menos 2 variables.")
        return
    
    st.session_state['cluster_selected_cols'] = selected_cols
    
    c1, c2 = st.columns(2)
    with c1:
        st.slider("K máximo a explorar:", 3, 15, 10, key='cluster_max_k')
    
    with c2:
        sample_size = st.number_input(
            "Máximo de muestras (0 = todas):", 
            0, len(df), min(5000, len(df)),
            key='cluster_sample'
        )
        if sample_size == 0:
            sample_size = len(df)
        st.session_state['cluster_sample_size'] = sample_size
    
    st.radio(
        "Método de visualización 2D:", 
        ["PCA", "t-SNE"],
        horizontal=True, key='cluster_viz_method',
        help="PCA: rápido y lineal. t-SNE: no-lineal, mejor para datasets complejos (más lento)."
    )
    
    # Info
    max_k = st.session_state.get('cluster_max_k', 10)
    n_sample = min(sample_size, len(df))
    st.markdown(f"""
    <div class="insight-box">
    ℹ️ Se usarán <b>{n_sample:,}</b> muestras × <b>{len(selected_cols)}</b> variables.
    Los datos serán normalizados con <b>StandardScaler</b> antes del clustering.
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("✅ Preparar Datos para Clustering", key='btn_prep_cluster', type='primary'):
        with st.spinner("Preparando datos..."):
            sample_df = df.sample(min(n_sample, len(df)), random_state=42) if n_sample < len(df) else df
            X_scaled = _prepare_cluster_data(sample_df, meta, selected_cols)
            if X_scaled is not None:
                st.session_state['X_cluster'] = X_scaled
                st.session_state['df_cluster'] = sample_df.reset_index(drop=True)
                st.session_state['cluster_data_ready'] = True
                st.success(f"✅ Datos listos: {X_scaled.shape[0]:,} × {X_scaled.shape[1]}")


# ─── Tab: Selección de K (Elbow) ──────────────────────────────────────────────
def _tab_elbow(df: pd.DataFrame, meta: Dict) -> None:
    
    if not st.session_state.get('cluster_data_ready', False):
        st.info("⚙️ Prepara los datos en la pestaña 'Configuración' primero.")
        return
    
    X = st.session_state.get('X_cluster')
    max_k = st.session_state.get('cluster_max_k', 10)
    
    if st.button("📊 Calcular Método del Codo + Silhouette", key='btn_elbow', type='primary'):
        with st.spinner("Calculando inertias y silhouette scores..."):
            k_range, inertias, silhouettes = _compute_elbow(X, max_k)
            if k_range:
                st.session_state['elbow_k_range'] = k_range
                st.session_state['elbow_inertias'] = inertias
                st.session_state['elbow_silhouettes'] = silhouettes
    
    if st.session_state.get('elbow_k_range'):
        k_range     = st.session_state['elbow_k_range']
        inertias    = st.session_state['elbow_inertias']
        silhouettes = st.session_state['elbow_silhouettes']
        
        # Encontrar mejor K
        best_k_sil = k_range[np.argmax(silhouettes)]
        best_k_elbow = _find_elbow_k(k_range, inertias)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{best_k_sil}</div>
                <div class="metric-label">K óptimo (Silhouette)</div>
                <div class="metric-delta positive">Score: {max(silhouettes):.3f}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{best_k_elbow}</div>
                <div class="metric-label">K sugerido (Codo)</div>
                <div class="metric-delta positive">Método geométrico</div>
            </div>""", unsafe_allow_html=True)
        
        st.session_state['best_k_suggested'] = best_k_sil
        
        st.plotly_chart(
            elbow_chart(k_range, inertias, silhouettes, best_k=best_k_sil),
            use_container_width=True
        )
        
        # Tabla de scores
        k_df = pd.DataFrame({
            'K': k_range,
            'Inercia (WCSS)': [round(v, 2) for v in inertias],
            'Silhouette Score': [round(v, 4) for v in silhouettes],
        })
        st.dataframe(k_df, use_container_width=True, hide_index=True)
        
        # Explicación
        st.markdown(f"""
        <div class="insight-box">
        💡 <b>¿Qué K elegir?</b><br>
        • <b>Método del Codo (K={best_k_elbow})</b>: El punto donde la reducción de inercia 
          se "aplana" — agregar más clusters aporta poco.<br>
        • <b>Silhouette Score (K={best_k_sil})</b>: El K con mayor cohesión interna y separación 
          entre clusters. Rango [-1, 1], más alto = mejor.<br>
        • <b>Recomendación</b>: Usa K={best_k_sil} como punto de partida y valida con el 
          conocimiento del dominio.
        </div>
        """, unsafe_allow_html=True)


def _compute_elbow(X: np.ndarray, max_k: int):
    """Calcula inertias y silhouette scores para K de 2 a max_k."""
    k_range, inertias, silhouettes = [], [], []
    progress = st.progress(0)
    
    for i, k in enumerate(range(2, max_k + 1)):
        progress.progress((i + 1) / (max_k - 1))
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            sil = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=42)
            silhouettes.append(sil)
            k_range.append(k)
        except Exception:
            continue
    
    progress.empty()
    return k_range, inertias, silhouettes


def _find_elbow_k(k_range: List[int], inertias: List[float]) -> int:
    """Detecta el codo usando el método de la distancia máxima."""
    if len(k_range) < 3:
        return k_range[0] if k_range else 2
    
    # Normalizar
    n = len(k_range)
    x = np.array(k_range, dtype=float)
    y = np.array(inertias, dtype=float)
    x = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y = (y - y.min()) / (y.max() - y.min() + 1e-10)
    
    # Línea de inicio a fin
    line = np.array([x[-1] - x[0], y[-1] - y[0]])
    line = line / (np.linalg.norm(line) + 1e-10)
    
    # Distancia perpendicular
    distances = []
    for i in range(n):
        pt = np.array([x[i] - x[0], y[i] - y[0]])
        d = abs(np.cross(line, pt))
        distances.append(d)
    
    elbow_idx = np.argmax(distances)
    return k_range[elbow_idx]


# ─── Tab: K-Means ─────────────────────────────────────────────────────────────
def _tab_kmeans(df: pd.DataFrame, meta: Dict) -> None:
    
    if not st.session_state.get('cluster_data_ready', False):
        st.info("⚙️ Prepara los datos primero.")
        return
    
    X = st.session_state.get('X_cluster')
    best_k = st.session_state.get('best_k_suggested', 3)
    
    k = st.slider("Número de clusters K:", 2, 15, best_k, key='kmeans_k')
    
    c1, c2 = st.columns(2)
    with c1:
        init_method = st.selectbox("Método de inicialización:", ['k-means++', 'random'], key='kmeans_init')
        n_init = st.slider("Número de inicializaciones:", 5, 30, 10, key='kmeans_ninit')
    with c2:
        max_iter = st.slider("Iteraciones máximas:", 100, 1000, 300, key='kmeans_maxiter')
    
    if st.button("🚀 Entrenar K-Means", key='btn_kmeans', type='primary'):
        with st.spinner(f"Entrenando K-Means con K={k}..."):
            km = KMeans(n_clusters=k, init=init_method, n_init=n_init,
                        max_iter=max_iter, random_state=42)
            labels = km.fit_predict(X)
            
            sil_score = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=42)
            
            st.session_state.update({
                'kmeans_model':    km,
                'kmeans_labels':   labels,
                'kmeans_k_trained': k,
                'kmeans_sil':      sil_score,
                'kmeans_done':     True,
                'cluster_labels':  labels,
                'active_clustering': 'kmeans',
            })
            
            st.success(f"✅ K-Means entrenado | Clusters: {k} | Silhouette: {sil_score:.3f}")
    
    if st.session_state.get('kmeans_done', False):
        sil = st.session_state.get('kmeans_sil', 0)
        km  = st.session_state.get('kmeans_model')
        labels = st.session_state.get('kmeans_labels')
        k   = st.session_state.get('kmeans_k_trained', st.session_state.get('kmeans_k', 3))
        
        # Métricas
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{k}</div>
                <div class="metric-label">Clusters</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            color = '#10B981' if sil > 0.5 else '#F59E0B' if sil > 0.3 else '#EF4444'
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{color}">{sil:.4f}</div>
                <div class="metric-label">Silhouette Score</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            inertia = km.inertia_ if km else 0
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{inertia:,.0f}</div>
                <div class="metric-label">Inercia (WCSS)</div>
            </div>""", unsafe_allow_html=True)
        
        # Distribución de clusters
        unique, counts = np.unique(labels, return_counts=True)
        fig = go.Figure(go.Bar(
            x=[f'Cluster {u}' for u in unique],
            y=counts,
            marker_color=PALETTE_PRIMARY[:len(unique)],
            text=[f'{c} ({c/len(labels)*100:.1f}%)' for c in counts],
            textposition='outside',
        ))
        apply_layout(fig, 'Tamaño de Clusters', height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calidad
        if sil > 0.5:
            msg = f"✅ <b>Segmentación de buena calidad</b> (Silhouette={sil:.3f}). Los clusters están bien separados y son compactos."
        elif sil > 0.3:
            msg = f"⚠️ <b>Segmentación aceptable</b> (Silhouette={sil:.3f}). Considera explorar otros valores de K o más variables."
        else:
            msg = f"❌ <b>Segmentación pobre</b> (Silhouette={sil:.3f}). Los clusters se solapan. Prueba normalización diferente o más features."
        
        st.markdown(f'<div class="insight-box">{msg}</div>', unsafe_allow_html=True)


# ─── Tab: Jerárquico ──────────────────────────────────────────────────────────
def _tab_hierarchical(df: pd.DataFrame, meta: Dict) -> None:
    
    if not st.session_state.get('cluster_data_ready', False):
        st.info("⚙️ Prepara los datos primero.")
        return
    
    X = st.session_state.get('X_cluster')
    best_k = st.session_state.get('best_k_suggested', 3)
    
    c1, c2 = st.columns(2)
    with c1:
        k_hier = st.slider("Número de clusters:", 2, 15, best_k, key='hier_k')
        linkage_method = st.selectbox(
            "Método de enlace:", 
            ['ward', 'complete', 'average', 'single'],
            key='hier_linkage',
            help="ward: minimiza varianza. complete: máxima distancia. average: promedio."
        )
    with c2:
        max_dendrogram_samples = st.slider(
            "Muestras para dendrograma:", 
            10, min(300, len(X)), min(100, len(X)),
            key='hier_dend_n',
            help="El dendrograma se muestra para una muestra del dataset."
        )
    
    if st.button("🌳 Entrenar Clustering Jerárquico", key='btn_hier', type='primary'):
        with st.spinner("Entrenando clustering jerárquico..."):
            agg = AgglomerativeClustering(n_clusters=k_hier, linkage=linkage_method)
            labels_hier = agg.fit_predict(X)
            
            sil_hier = silhouette_score(X, labels_hier, sample_size=min(2000, len(X)), random_state=42)
            
            st.session_state.update({
                'hier_model':          agg,
                'hier_labels':         labels_hier,
                'hier_k_trained':      k_hier,
                'hier_sil':            sil_hier,
                'hier_linkage_trained': linkage_method,
                'hier_done':           True,
            })
            
            st.success(f"✅ Clustering Jerárquico | Clusters: {k_hier} | Silhouette: {sil_hier:.3f}")
    
    if st.session_state.get('hier_done', False):
        sil_hier = st.session_state.get('hier_sil', 0)
        k_hier   = st.session_state.get('hier_k_trained', st.session_state.get('hier_k', 3))
        linkage_method = st.session_state.get('hier_linkage_trained', st.session_state.get('hier_linkage', 'ward'))
        
        c1, c2 = st.columns(2)
        with c1:
            color = '#10B981' if sil_hier > 0.5 else '#F59E0B' if sil_hier > 0.3 else '#EF4444'
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{color}">{sil_hier:.4f}</div>
                <div class="metric-label">Silhouette Score (Jerárquico)</div>
            </div>""", unsafe_allow_html=True)
        
        # Dendrograma
        st.markdown("#### 🌳 Dendrograma")
        with st.spinner("Generando dendrograma..."):
            try:
                X_sample = X[:max_dendrogram_samples]
                Z = linkage(X_sample, method=linkage_method)
                
                fig_dend, ax = plt.subplots(figsize=(12, 5))
                fig_dend.patch.set_facecolor('#0F0F1A')
                ax.set_facecolor('#1A1A2E')
                
                dendrogram(
                    Z, ax=ax,
                    color_threshold=Z[-k_hier+1, 2],
                    above_threshold_color='#94A3B8',
                    leaf_font_size=8,
                )
                ax.set_title(f'Dendrograma — {linkage_method} linkage', 
                            color='#A78BFA', fontsize=13, pad=15)
                ax.set_xlabel('Índice de muestra', color='#94A3B8')
                ax.set_ylabel('Distancia', color='#94A3B8')
                ax.tick_params(colors='#94A3B8')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#374151')
                
                plt.tight_layout()
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=120, facecolor='#0F0F1A')
                buf.seek(0)
                st.image(buf, use_column_width=True)
                plt.close()
            except Exception as e:
                st.warning(f"⚠️ No se pudo generar el dendrograma: {e}")
        
        # Comparación K-Means vs Jerárquico
        if st.session_state.get('kmeans_done', False):
            sil_km = st.session_state.get('kmeans_sil', 0)
            st.markdown("#### ⚡ Comparación K-Means vs Jerárquico")
            
            comp_df = pd.DataFrame({
                'Método': ['K-Means', 'Clustering Jerárquico'],
                'Silhouette Score': [round(sil_km, 4), round(sil_hier, 4)],
                'K': [st.session_state.get('kmeans_k_trained', st.session_state.get('kmeans_k', '-')), k_hier],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            winner = 'K-Means' if sil_km >= sil_hier else 'Clustering Jerárquico'
            st.markdown(f'<div class="insight-box">🏆 <b>Mejor segmentación:</b> {winner} con Silhouette = {max(sil_km, sil_hier):.4f}</div>',
                       unsafe_allow_html=True)


# ─── Tab: Visualización ───────────────────────────────────────────────────────
def _tab_viz_cluster(df: pd.DataFrame, meta: Dict) -> None:
    
    if not (st.session_state.get('kmeans_done') or st.session_state.get('hier_done')):
        st.info("⚙️ Entrena al menos un modelo de clustering primero.")
        return
    
    X = st.session_state.get('X_cluster')
    viz_method = st.session_state.get('cluster_viz_method', 'PCA')
    
    # Seleccionar qué labels usar
    options = []
    if st.session_state.get('kmeans_done'): options.append('K-Means')
    if st.session_state.get('hier_done'):   options.append('Jerárquico')
    
    model_sel = st.radio("Visualizar clusters de:", options, horizontal=True, key='viz_model_sel')
    
    if model_sel == 'K-Means':
        labels = st.session_state.get('kmeans_labels')
        centroids_raw = st.session_state.get('kmeans_model').cluster_centers_ if st.session_state.get('kmeans_model') else None
    else:
        labels = st.session_state.get('hier_labels')
        centroids_raw = None
    
    if labels is None:
        return
    
    if st.button(f"🗺️ Visualizar con {viz_method}", key='btn_viz_cluster', type='primary'):
        with st.spinner(f"Reduciendo dimensiones con {viz_method}..."):
            try:
                if viz_method == 'PCA':
                    reducer = PCA(n_components=2, random_state=42)
                    components = reducer.fit_transform(X)
                    expl_var = reducer.explained_variance_ratio_
                    extra_info = f"Varianza explicada: PC1={expl_var[0]:.1%}, PC2={expl_var[1]:.1%}"
                    centroids_2d = reducer.transform(centroids_raw) if centroids_raw is not None else None
                else:
                    perplexity = min(30, len(X) - 1)
                    reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                    components = reducer.fit_transform(X)
                    extra_info = "t-SNE (visualización no lineal)"
                    centroids_2d = None
                
                st.session_state['cluster_2d'] = components
                st.session_state['cluster_2d_info'] = extra_info
                st.session_state['cluster_centroids_2d'] = centroids_2d
                
                st.plotly_chart(
                    cluster_scatter(components, labels, method=viz_method, 
                                   centroids=centroids_2d),
                    use_container_width=True
                )
                st.markdown(f'<div class="insight-box">ℹ️ {extra_info}</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Error en visualización: {str(e)}")
    
    elif st.session_state.get('cluster_2d') is not None:
        components = st.session_state['cluster_2d']
        centroids_2d = st.session_state.get('cluster_centroids_2d')
        st.plotly_chart(
            cluster_scatter(components, labels, method=viz_method, centroids=centroids_2d),
            use_container_width=True
        )
        st.markdown(f'<div class="insight-box">ℹ️ {st.session_state.get("cluster_2d_info", "")}</div>',
                   unsafe_allow_html=True)


# ─── Tab: Perfiles de Clusters ────────────────────────────────────────────────
def _tab_profiles(df: pd.DataFrame, meta: Dict) -> None:
    
    if not (st.session_state.get('kmeans_done') or st.session_state.get('hier_done')):
        st.info("⚙️ Entrena al menos un modelo de clustering primero.")
        return
    
    df_cluster = st.session_state.get('df_cluster', df)
    selected_cols = st.session_state.get('cluster_cols', meta.get('numeric_cols', []))
    
    options = []
    if st.session_state.get('kmeans_done'): options.append('K-Means')
    if st.session_state.get('hier_done'):   options.append('Jerárquico')
    
    model_sel = st.radio("Perfiles de:", options, horizontal=True, key='profile_model_sel')
    
    if model_sel == 'K-Means':
        labels = st.session_state.get('kmeans_labels')
    else:
        labels = st.session_state.get('hier_labels')
    
    if labels is None or len(labels) != len(df_cluster):
        st.warning("⚠️ Las etiquetas no coinciden con el DataFrame. Re-entrena el modelo.")
        return
    
    df_prof = df_cluster[selected_cols].copy()
    df_prof['Cluster'] = labels
    
    # Perfil de medias
    profile = df_prof.groupby('Cluster')[selected_cols].agg(['mean', 'std', 'count'])
    profile.columns = ['_'.join(c) for c in profile.columns]
    
    mean_cols = [c for c in profile.columns if c.endswith('_mean')]
    
    # Tabla de medias por cluster
    st.markdown("#### 📊 Medias por Cluster")
    mean_df = df_prof.groupby('Cluster')[selected_cols].mean().round(3)
    mean_df['N_miembros'] = df_prof.groupby('Cluster').size()
    st.dataframe(mean_df, use_container_width=True)
    
    # Heatmap de perfiles
    if len(selected_cols) >= 2:
        mean_norm = (mean_df[selected_cols] - mean_df[selected_cols].min()) / (
            mean_df[selected_cols].max() - mean_df[selected_cols].min() + 1e-10
        )
        
        fig = go.Figure(go.Heatmap(
            z=mean_norm.values,
            x=selected_cols,
            y=[f'Cluster {i}' for i in mean_norm.index],
            colorscale='RdBu_r',
            text=mean_df[selected_cols].round(2).values.astype(str),
            texttemplate='%{text}',
            colorbar=dict(title='Valor\nNormalizado'),
            hovertemplate='Variable: %{x}<br>Cluster: %{y}<br>Media: %{text}<extra></extra>',
        ))
        apply_layout(fig, 'Heatmap de Perfiles por Cluster', height=max(300, len(mean_df)*70+80))
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    
    # Interpretación automática
    st.markdown("#### 💬 Interpretación Automática")
    interp = interpret_clusters(mean_df.reset_index(), len(mean_df), selected_cols, model_sel)
    st.markdown(interp)
    
    # Guardar info para export
    st.session_state['cluster_profiles'] = mean_df
    st.session_state['cluster_info'] = {
        'method': model_sel,
        'k': len(mean_df),
        'silhouette': st.session_state.get('kmeans_sil' if model_sel == 'K-Means' else 'hier_sil', 0)
    }
