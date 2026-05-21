"""
export.py
=========
Exportación de reportes, métricas, gráficos y datasets procesados.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import io
import zipfile
import json
from datetime import datetime


# ─── Sección Principal ────────────────────────────────────────────────────────
def render_export(df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📤</div>
        <div>
            <div class="section-title">Exportación de Resultados</div>
            <div class="section-subtitle">Descarga reportes, métricas, datasets y gráficos</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs([
        "📊 Métricas (Excel)", "💾 Dataset Procesado",
        "📄 Reporte PDF", "📋 Resumen JSON"
    ])
    
    with tabs[0]: _tab_export_metrics(df, meta)
    with tabs[1]: _tab_export_dataset(df, meta)
    with tabs[2]: _tab_export_pdf(df, meta)
    with tabs[3]: _tab_export_json(df, meta)


# ─── Tab: Exportar Métricas ───────────────────────────────────────────────────
def _tab_export_metrics(df: pd.DataFrame, meta: Dict) -> None:
    
    st.markdown("### 📊 Exportar Métricas a Excel")
    
    results_df = st.session_state.get('model_results_df')
    baseline   = st.session_state.get('baseline_metrics', {})
    test_metrics = st.session_state.get('test_metrics', {})
    
    if results_df is None and not baseline:
        st.info("⚙️ Entrena modelos para exportar métricas.")
        return
    
    # Construir Excel multi-hoja
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Estilos
        header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#7C3AED', 'font_color': 'white',
            'border': 1, 'align': 'center',
        })
        cell_fmt = workbook.add_format({'border': 1, 'align': 'center'})
        num_fmt  = workbook.add_format({'border': 1, 'align': 'center', 'num_format': '0.0000'})
        title_fmt = workbook.add_format({
            'bold': True, 'font_size': 14, 'font_color': '#7C3AED',
        })
        
        # Hoja 1: Dataset Info
        ws_info = workbook.add_worksheet('Dataset Info')
        ws_info.write(0, 0, 'Información del Dataset', title_fmt)
        info_rows = [
            ['Filas', meta.get('n_rows', 0)],
            ['Columnas', meta.get('n_cols', 0)],
            ['Variables Numéricas', len(meta.get('numeric_cols', []))],
            ['Variables Categóricas', len(meta.get('categorical_cols', []))],
            ['Valores Nulos Total', sum(meta.get('null_counts', {}).values())],
            ['Memoria (MB)', meta.get('memory_mb', 0)],
            ['Fecha Análisis', datetime.now().strftime('%Y-%m-%d %H:%M')],
        ]
        for i, (label, val) in enumerate(info_rows, 2):
            ws_info.write(i, 0, label)
            ws_info.write(i, 1, val)
        ws_info.set_column(0, 0, 25)
        ws_info.set_column(1, 1, 20)
        
        # Hoja 2: Comparación de modelos
        if results_df is not None and not results_df.empty:
            results_df.to_excel(writer, sheet_name='Comparación Modelos', index=False)
            ws = writer.sheets['Comparación Modelos']
            for col_num, col_name in enumerate(results_df.columns):
                ws.write(0, col_num, col_name, header_fmt)
                ws.set_column(col_num, col_num, max(15, len(str(col_name)) + 5))
        
        # Hoja 3: Métricas de test
        if test_metrics:
            test_df = pd.DataFrame([{
                'Modelo Final': st.session_state.get('best_model_final', 'N/A'),
                'Accuracy': test_metrics.get('accuracy', 0),
                'Precision': test_metrics.get('precision', 0),
                'Recall': test_metrics.get('recall', 0),
                'F1-Score': test_metrics.get('f1', 0),
                'AUC-ROC': test_metrics.get('auc', 0),
                'Evaluado en': 'Test Set',
                'Fecha': datetime.now().strftime('%Y-%m-%d %H:%M'),
            }])
            test_df.to_excel(writer, sheet_name='Métricas Test Final', index=False)
        
        # Hoja 4: Estadísticas descriptivas
        num_cols = meta.get('numeric_cols', [])
        if num_cols:
            desc = df[num_cols].describe().T
            desc.index.name = 'Variable'
            desc.to_excel(writer, sheet_name='Estadísticas Descriptivas')
        
        # Hoja 5: Perfiles de clusters
        cluster_profiles = st.session_state.get('cluster_profiles')
        if cluster_profiles is not None:
            cluster_profiles.to_excel(writer, sheet_name='Perfiles Clusters')
    
    output.seek(0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    st.download_button(
        label="⬇️ Descargar Métricas (Excel)",
        data=output.getvalue(),
        file_name=f"ml_metricas_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key='download_excel_metrics'
    )
    
    # Previsualización
    if results_df is not None:
        st.markdown("**Vista previa — Comparación de Modelos:**")
        st.dataframe(results_df, use_container_width=True, hide_index=True)


# ─── Tab: Exportar Dataset ────────────────────────────────────────────────────
def _tab_export_dataset(df: pd.DataFrame, meta: Dict) -> None:
    
    st.markdown("### 💾 Exportar Dataset")
    
    export_opt = st.radio(
        "¿Qué exportar?",
        ["Dataset original", "Dataset procesado (features)", "Dataset con etiquetas de cluster"],
        key='export_ds_type'
    )
    
    df_to_export = None
    filename = "dataset"
    
    if export_opt == "Dataset original":
        df_to_export = df.copy()
        filename = "dataset_original"
        st.dataframe(df.head(10), use_container_width=True)
    
    elif export_opt == "Dataset procesado (features)":
        X_proc = st.session_state.get('X_processed')
        feature_names = st.session_state.get('feature_names', [])
        y = st.session_state.get('y_processed')
        
        if X_proc is None:
            st.warning("⚠️ No hay dataset procesado. Aplica el Preprocesamiento primero.")
            return
        
        if hasattr(X_proc, 'toarray'):
            data = X_proc.toarray()
        else:
            data = X_proc
        
        df_to_export = pd.DataFrame(data, columns=feature_names if feature_names else None)
        
        target_col = st.session_state.get('target_col')
        if y is not None and target_col:
            df_to_export[target_col] = y.values if hasattr(y, 'values') else y
        
        filename = "dataset_procesado"
        st.dataframe(df_to_export.head(10), use_container_width=True)
    
    elif export_opt == "Dataset con etiquetas de cluster":
        df_cluster = st.session_state.get('df_cluster', df)
        kmeans_labels = st.session_state.get('kmeans_labels')
        hier_labels   = st.session_state.get('hier_labels')
        
        if kmeans_labels is None and hier_labels is None:
            st.warning("⚠️ No hay clustering realizado.")
            return
        
        df_to_export = df_cluster.copy()
        if kmeans_labels is not None and len(kmeans_labels) == len(df_to_export):
            df_to_export['cluster_kmeans'] = kmeans_labels
        if hier_labels is not None and len(hier_labels) == len(df_to_export):
            df_to_export['cluster_jerarquico'] = hier_labels
        
        filename = "dataset_con_clusters"
        st.dataframe(df_to_export.head(10), use_container_width=True)
    
    if df_to_export is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df_to_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar CSV",
                data=csv_data,
                file_name=f"{filename}_{timestamp}.csv",
                mime="text/csv",
                key='download_csv_ds'
            )
        
        with col2:
            excel_out = io.BytesIO()
            df_to_export.to_excel(excel_out, index=False, engine='openpyxl')
            excel_out.seek(0)
            st.download_button(
                label="⬇️ Descargar Excel",
                data=excel_out.getvalue(),
                file_name=f"{filename}_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='download_excel_ds'
            )
        
        st.markdown(f"**Shape:** `{df_to_export.shape}`")


# ─── Tab: Reporte PDF ─────────────────────────────────────────────────────────
def _tab_export_pdf(df: pd.DataFrame, meta: Dict) -> None:
    
    st.markdown("### 📄 Reporte PDF Ejecutivo")
    
    st.markdown("""
    <div class="insight-box">
    📄 El reporte PDF incluirá:<br>
    • Información del dataset<br>
    • Estadísticas descriptivas<br>
    • Resultados de clustering (si disponible)<br>
    • Métricas de clasificación (si disponible)<br>
    • Conclusiones automáticas y recomendaciones
    </div>
    """, unsafe_allow_html=True)
    
    report_title = st.text_input("Título del reporte:", 
                                  value="Análisis de Machine Learning — Reporte Ejecutivo",
                                  key='pdf_title')
    author_name  = st.text_input("Autor:", value="ML Platform", key='pdf_author')
    
    if st.button("📄 Generar Reporte PDF", key='btn_pdf', type='primary'):
        with st.spinner("Generando PDF..."):
            try:
                pdf_bytes = _generate_pdf(df, meta, report_title, author_name)
                if pdf_bytes:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        label="⬇️ Descargar Reporte PDF",
                        data=pdf_bytes,
                        file_name=f"reporte_ml_{timestamp}.pdf",
                        mime="application/pdf",
                        key='download_pdf'
                    )
                    st.success("✅ PDF generado exitosamente.")
            except Exception as e:
                st.error(f"❌ Error generando PDF: {str(e)}")
                _generate_text_report_fallback(df, meta, report_title, author_name)


def _generate_pdf(df: pd.DataFrame, meta: Dict, title: str, author: str) -> Optional[bytes]:
    """Genera un reporte PDF usando fpdf2."""
    try:
        from fpdf import FPDF, XPos, YPos
    except ImportError:
        try:
            from fpdf2 import FPDF
        except ImportError:
            return None
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Colores corporativos
    PRIMARY = (124, 58, 237)
    DARK    = (15, 15, 26)
    GRAY    = (100, 116, 139)
    
    # Fondo
    pdf.set_fill_color(*DARK)
    pdf.rect(0, 0, 210, 297, 'F')
    
    # Header
    pdf.set_fill_color(*PRIMARY)
    pdf.rect(0, 0, 210, 45, 'F')
    
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(12)
    pdf.cell(0, 10, title[:60], align='C')
    
    pdf.set_font('Helvetica', '', 11)
    pdf.set_y(26)
    pdf.cell(0, 8, f'Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")} | Autor: {author}', align='C')
    
    pdf.set_y(55)
    
    # Sección: Dataset
    _pdf_section(pdf, '1. RESUMEN DEL DATASET', PRIMARY)
    
    info_rows = [
        ('Filas', f"{meta.get('n_rows', 0):,}"),
        ('Columnas', str(meta.get('n_cols', 0))),
        ('Variables Numéricas', str(len(meta.get('numeric_cols', [])))),
        ('Variables Categóricas', str(len(meta.get('categorical_cols', [])))),
        ('Valores Nulos', str(sum(meta.get('null_counts', {}).values()))),
        ('Memoria', f"{meta.get('memory_mb', 0)} MB"),
    ]
    _pdf_table(pdf, ['Métrica', 'Valor'], info_rows, PRIMARY)
    
    # Sección: Estadísticas
    num_cols = meta.get('numeric_cols', [])
    if num_cols:
        _pdf_section(pdf, '2. ESTADÍSTICAS DESCRIPTIVAS', PRIMARY)
        desc = df[num_cols[:6]].describe().round(3)
        
        headers = ['Variable'] + desc.columns.tolist()
        rows = []
        for idx in desc.index:
            row = [str(idx)] + [str(v) for v in desc.loc[idx].values]
            rows.append(row)
        _pdf_table(pdf, headers, rows, PRIMARY)
    
    # Sección: Modelos
    results_df = st.session_state.get('model_results_df')
    if results_df is not None and not results_df.empty:
        _pdf_section(pdf, '3. COMPARACIÓN DE MODELOS', PRIMARY)
        
        cols_to_show = ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        rows_m = []
        for _, row in results_df[cols_to_show].iterrows():
            rows_m.append([str(v) for v in row.values])
        _pdf_table(pdf, cols_to_show, rows_m, PRIMARY)
    
    # Sección: Clustering
    cluster_info = st.session_state.get('cluster_info')
    if cluster_info:
        _pdf_section(pdf, '4. SEGMENTACIÓN (CLUSTERING)', PRIMARY)
        
        sil = cluster_info.get('silhouette', 0)
        quality = "Buena" if sil > 0.5 else "Moderada" if sil > 0.3 else "Baja"
        
        cluster_rows = [
            ('Método', cluster_info.get('method', 'K-Means')),
            ('Clusters (K)', str(cluster_info.get('k', 0))),
            ('Silhouette Score', f"{sil:.4f}"),
            ('Calidad', quality),
        ]
        _pdf_table(pdf, ['Parámetro', 'Valor'], cluster_rows, PRIMARY)
    
    # Sección: Métricas Test
    test_metrics = st.session_state.get('test_metrics')
    if test_metrics:
        _pdf_section(pdf, '5. EVALUACIÓN FINAL EN TEST', PRIMARY)
        
        test_rows = [
            ('Accuracy',  f"{test_metrics.get('accuracy', 0):.4f}"),
            ('Precision', f"{test_metrics.get('precision', 0):.4f}"),
            ('Recall',    f"{test_metrics.get('recall', 0):.4f}"),
            ('F1-Score',  f"{test_metrics.get('f1', 0):.4f}"),
            ('AUC-ROC',   f"{test_metrics.get('auc', 0):.4f}"),
        ]
        _pdf_table(pdf, ['Métrica', 'Valor (Test)'], test_rows, PRIMARY)
    
    # Sección: Gráficos
    pdf_figures = st.session_state.get('pdf_figures', {})
    if pdf_figures:
        _pdf_section(pdf, '6. ANEXO: GRÁFICOS GENERADOS', PRIMARY)
        import tempfile
        import os
        for fig_title, fig in pdf_figures.items():
            try:
                img_bytes = fig.to_image(format="png", engine="kaleido", width=800, height=500)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name
                
                pdf.add_page()
                pdf.set_font('Helvetica', 'B', 12)
                pdf.set_text_color(*PRIMARY)
                pdf.cell(0, 10, fig_title[:80], align='C')
                pdf.ln(10)
                
                pdf.image(tmp_path, x=15, w=180)
                os.remove(tmp_path)
            except Exception as e:
                pdf.set_font('Helvetica', 'I', 10)
                pdf.set_text_color(*GRAY)
                pdf.cell(0, 10, f'(No se pudo exportar gráfico: {str(e)})', align='C')
                pdf.ln(5)
    
    # Footer
    pdf.set_y(-20)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 6, 'Generado por ML & Data Science Platform | Reporte Automático', align='C')
    
    return bytes(pdf.output())


def _pdf_section(pdf, title: str, color: tuple) -> None:
    """Agrega encabezado de sección al PDF."""
    from fpdf import FPDF
    pdf.ln(4)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(*color)
    pdf.cell(0, 8, title)
    pdf.ln(2)
    pdf.set_draw_color(*color)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_text_color(226, 232, 240)


def _pdf_table(pdf, headers: list, rows: list, color: tuple) -> None:
    """Genera tabla en el PDF."""
    col_w = 180 / len(headers)
    
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    for h in headers:
        pdf.cell(col_w, 7, str(h)[:20], border=1, align='C', fill=True)
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 8)
    for i, row in enumerate(rows):
        if i % 2 == 0:
            pdf.set_fill_color(26, 26, 46)
        else:
            pdf.set_fill_color(20, 20, 40)
        pdf.set_text_color(226, 232, 240)
        for cell in row:
            pdf.cell(col_w, 6, str(cell)[:20], border=1, align='C', fill=True)
        pdf.ln()
    pdf.ln(3)


def _generate_text_report_fallback(df, meta, title, author):
    """Genera reporte de texto plano como alternativa."""
    lines = [
        f"# {title}",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Autor: {author}",
        "",
        "## RESUMEN DEL DATASET",
        f"- Filas: {meta.get('n_rows', 0):,}",
        f"- Columnas: {meta.get('n_cols', 0)}",
        f"- Variables Numéricas: {len(meta.get('numeric_cols', []))}",
        f"- Variables Categóricas: {len(meta.get('categorical_cols', []))}",
        f"- Nulos Total: {sum(meta.get('null_counts', {}).values())}",
        "",
    ]
    
    results_df = st.session_state.get('model_results_df')
    if results_df is not None:
        lines.append("## COMPARACIÓN DE MODELOS")
        lines.append(results_df.to_string(index=False))
    
    text = "\n".join(lines)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    st.download_button(
        label="⬇️ Descargar Reporte (TXT)",
        data=text.encode('utf-8'),
        file_name=f"reporte_ml_{timestamp}.txt",
        mime="text/plain",
        key='download_txt_fallback'
    )


# ─── Tab: Resumen JSON ────────────────────────────────────────────────────────
def _tab_export_json(df: pd.DataFrame, meta: Dict) -> None:
    
    st.markdown("### 📋 Exportar Configuración y Resultados (JSON)")
    
    # Construir JSON de resultados
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'n_rows':          meta.get('n_rows', 0),
            'n_cols':          meta.get('n_cols', 0),
            'numeric_cols':    meta.get('numeric_cols', []),
            'categorical_cols': meta.get('categorical_cols', []),
            'null_counts':     meta.get('null_counts', {}),
        },
        'preprocessing': {
            'target_col':      st.session_state.get('target_col'),
            'scaler':          st.session_state.get('scaler_type'),
            'encoding':        st.session_state.get('encoding_type'),
            'imputation_num':  st.session_state.get('num_impute'),
        },
        'models': {},
        'clustering': {},
    }
    
    # Modelos
    results_df = st.session_state.get('model_results_df')
    if results_df is not None:
        for _, row in results_df.iterrows():
            summary['models'][row['Modelo']] = {
                k: v for k, v in row.items() if k != 'Modelo'
            }
    
    test_m = st.session_state.get('test_metrics')
    if test_m:
        summary['final_test'] = {k: v for k, v in test_m.items() 
                                  if not isinstance(v, np.ndarray)}
    
    # Clustering
    cluster_info = st.session_state.get('cluster_info')
    if cluster_info:
        summary['clustering'] = cluster_info
    
    # Convertir numpy types
    def convert_types(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert_types(v) for v in obj]
        return obj
    
    summary = convert_types(summary)
    json_str = json.dumps(summary, indent=2, ensure_ascii=False)
    
    st.json(summary)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    st.download_button(
        label="⬇️ Descargar JSON",
        data=json_str.encode('utf-8'),
        file_name=f"resultados_ml_{timestamp}.json",
        mime="application/json",
        key='download_json'
    )
