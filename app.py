"""
app.py
ClinVal Biostats — Lean Analytical Interface.
Imports untouched `stats_engine.py` and standalone `clsi_ep05_engine.py`.
Displays core metrics, statistical matrices, and visualizations only (no report/certificate generation).
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 1. Untouched stats_engine.py imports
from stats_engine import (
    bland_altman_analysis,
    compute_roc_and_cutoffs,
    delong_roc_ci,
    deming_regression,
    estimate_mdp_bias,
    evaluate_cutoff,
    generate_demo_dataset,
    paired_delong_test,
    passing_bablok_regression,
    run_clinical_etl,
)

# 2. Standalone CLSI EP05-A3 engine imports
from clsi_ep05_engine import (
    PrecisionLevelResult,
    compute_clsi_ep05_precision,
    format_precision_summary_table,
    generate_ep05_demo_dataset,
)

st.set_page_config(
    page_title="ClinVal Biostats | Analytical Metrics & Matrices",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_CONFIG = {"displaylogo": False}


# ==============================================================================
# VISUALIZATION BUILDERS
# ==============================================================================

def build_roc_figure(
    roc_df: pd.DataFrame,
    delong_res,
    cutoffs: Dict[str, float],
    active_threshold: float,
    comp_roc_df: Optional[pd.DataFrame] = None,
    comp_delong_res=None,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Reference Line (AUC = 0.50)",
            line=dict(color="#94a3b8", dash="dash", width=1.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=roc_df["FPR"],
            y=roc_df["Sensitivity"],
            mode="lines",
            name=f"Index Assay (AUC = {delong_res.auc:.3f}, 95% CI: {delong_res.ci_lower:.3f}–{delong_res.ci_upper:.3f})",
            line=dict(color="#0284c7", width=3),
            customdata=np.stack([roc_df["Threshold"], roc_df["Specificity"]], axis=-1),
            hovertemplate=(
                "Cutoff: %{customdata[0]:.3f}<br>"
                "Sensitivity: %{y:.1%}<br>"
                "Specificity: %{customdata[1]:.1%}<extra></extra>"
            ),
        )
    )

    if comp_roc_df is not None and comp_delong_res is not None:
        fig.add_trace(
            go.Scatter(
                x=comp_roc_df["FPR"],
                y=comp_roc_df["Sensitivity"],
                mode="lines",
                name=f"Comparator Assay (AUC = {comp_delong_res.auc:.3f}, 95% CI: {comp_delong_res.ci_lower:.3f}–{comp_delong_res.ci_upper:.3f})",
                line=dict(color="#d97706", width=2.2, dash="dot"),
            )
        )

    marker_styles = {
        "Youden's J (Balanced)": ("diamond", "#16a34a"),
        "Closest-to-(0,1)": ("circle", "#7c3aed"),
        "Rule-Out (Sens ≥ 95%)": ("triangle-up", "#0d9488"),
        "Rule-In (Spec ≥ 95%)": ("square", "#dc2626"),
    }

    for label, thresh_val in cutoffs.items():
        idx = int((roc_df["Threshold"] - thresh_val).abs().idxmin())
        symbol, color = marker_styles.get(label, ("circle", "#0f172a"))
        fig.add_trace(
            go.Scatter(
                x=[roc_df.loc[idx, "FPR"]],
                y=[roc_df.loc[idx, "Sensitivity"]],
                mode="markers",
                name=f"{label}: {thresh_val:.2f}",
                marker=dict(size=11, symbol=symbol, color=color, line=dict(width=1.5, color="white")),
            )
        )

    act_idx = int((roc_df["Threshold"] - active_threshold).abs().idxmin())
    fig.add_trace(
        go.Scatter(
            x=[roc_df.loc[act_idx, "FPR"]],
            y=[roc_df.loc[act_idx, "Sensitivity"]],
            mode="markers",
            name=f"Active Cutoff ({active_threshold:.2f})",
            marker=dict(size=16, symbol="circle-open", color="#0f172a", line=dict(width=2.5)),
        )
    )

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="1 − Specificity (False Positive Rate)",
        yaxis_title="Sensitivity (True Positive Rate)",
        template="plotly_white",
        height=490,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.85)"),
        xaxis=dict(range=[-0.02, 1.02], constrain="domain"),
        yaxis=dict(range=[-0.02, 1.02], scaleanchor="x", scaleratio=1),
    )
    return fig


def build_method_comparison_figure(x: np.ndarray, y: np.ndarray, pb_res, dem_res, x_label: str, y_label: str) -> go.Figure:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    grid = np.linspace(max(0.0, x_min * 0.9), x_max * 1.05, 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=f"Samples (N = {len(x)})", marker=dict(color="#334155", size=7, opacity=0.72)))
    fig.add_trace(go.Scatter(x=grid, y=grid, mode="lines", name="Identity (y = x)", line=dict(color="#94a3b8", dash="dash", width=1.5)))

    pb_y = pb_res.intercept + pb_res.slope * grid
    pb_low = pb_res.intercept_ci[0] + pb_res.slope_ci[0] * grid
    pb_high = pb_res.intercept_ci[1] + pb_res.slope_ci[1] * grid

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([grid, grid[::-1]]),
            y=np.concatenate([pb_high, pb_low[::-1]]),
            fill="toself",
            fillcolor="rgba(2, 132, 199, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Passing-Bablok 95% CI",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid, y=pb_y,
            mode="lines",
            name=f"Passing-Bablok: y = {pb_res.slope:.3f}x + {pb_res.intercept:.2f}",
            line=dict(color="#0284c7", width=2.8),
        )
    )

    dem_y = dem_res.intercept + dem_res.slope * grid
    fig.add_trace(
        go.Scatter(
            x=grid, y=dem_y,
            mode="lines",
            name=f"Deming (λ={dem_res.error_ratio:.2f}): y = {dem_res.slope:.3f}x + {dem_res.intercept:.2f}",
            line=dict(color="#dc2626", width=2.2, dash="dot"),
        )
    )

    fig.update_layout(
        title="Passing-Bablok & Deming Method Comparison",
        xaxis_title=f"Reference Standard ({x_label})",
        yaxis_title=f"Index Assay ({y_label})",
        template="plotly_white",
        height=460,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


def build_bland_altman_figure(ba_res) -> go.Figure:
    unit_suffix = " (%)" if ba_res.mode == "percentage" else " (Absolute)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ba_res.means, y=ba_res.diffs, mode="markers", name="Paired Differences", marker=dict(color="#0f172a", size=7, opacity=0.7)))
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")

    fig.add_hrect(y0=ba_res.mean_bias_ci[0], y1=ba_res.mean_bias_ci[1], fillcolor="rgba(22, 163, 74, 0.14)", line_width=0)
    fig.add_hline(y=ba_res.mean_bias, line_color="#16a34a", line_width=2.5, annotation_text=f"Mean Bias: {ba_res.mean_bias:.2f}", annotation_position="top right")

    for loa_val, loa_ci, label in [
        (ba_res.loa_upper, ba_res.loa_upper_ci, f"+1.96 SD: {ba_res.loa_upper:.2f}"),
        (ba_res.loa_lower, ba_res.loa_lower_ci, f"-1.96 SD: {ba_res.loa_lower:.2f}"),
    ]:
        fig.add_hrect(y0=loa_ci[0], y1=loa_ci[1], fillcolor="rgba(220, 38, 38, 0.10)", line_width=0)
        fig.add_hline(y=loa_val, line_color="#dc2626", line_dash="dash", line_width=2, annotation_text=label, annotation_position="bottom right" if loa_val < 0 else "top right")

    fig.update_layout(
        title=f"Bland-Altman Agreement Plot{unit_suffix}",
        xaxis_title="Mean of Index & Reference Methods",
        yaxis_title=f"Difference (Index − Reference){unit_suffix}",
        template="plotly_white",
        height=460,
    )
    return fig


def build_precision_trajectory_figure(prec_res: PrecisionLevelResult) -> go.Figure:
    df_rec = prec_res.clean_records
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_rec["Day"],
            y=df_rec["Value"],
            mode="markers",
            name="Replicates",
            marker=dict(color="#475569", size=7, opacity=0.72),
        )
    )

    daily = df_rec.groupby("Day", sort=False)["Value"].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=daily["Day"],
            y=daily["Value"],
            mode="lines+markers",
            name="Daily Mean",
            line=dict(color="#0284c7", width=2.4),
            marker=dict(symbol="diamond", size=9, color="#0284c7"),
        )
    )

    mean_v = prec_res.mean
    s_wl = prec_res.s_wl
    fig.add_hline(y=mean_v, line_color="#16a34a", line_width=2.0, annotation_text=f"Mean: {mean_v:.3f}", annotation_position="top left")
    for mult, color, dash_style in [(2.0, "#d97706", "dash"), (3.0, "#dc2626", "dot")]:
        fig.add_hline(y=mean_v + mult * s_wl, line_color=color, line_dash=dash_style, line_width=1.4, annotation_text=f"+{int(mult)} S_WL", annotation_position="top right")
        fig.add_hline(y=mean_v - mult * s_wl, line_color=color, line_dash=dash_style, line_width=1.4, annotation_text=f"−{int(mult)} S_WL", annotation_position="bottom right")

    fig.update_layout(
        title=f"Multi-Day QC Trajectory — {prec_res.qc_level} (CV_WL = {prec_res.cv_wl:.2f}%)",
        xaxis_title="Analytical Day",
        yaxis_title="Measured Value",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def build_variance_components_figure(prec_results: List[PrecisionLevelResult]) -> go.Figure:
    levels = [r.qc_level for r in prec_results]
    pct_r = [r.pct_var_r for r in prec_results]
    pct_run = [r.pct_var_run for r in prec_results]
    pct_day = [r.pct_var_day for r in prec_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=levels, x=pct_r, name="Within-Run (V_r)", orientation="h", marker_color="#0284c7", text=[f"{v:.1f}%" for v in pct_r], textposition="inside"))
    if any(v > 0 for v in pct_run):
        fig.add_trace(go.Bar(y=levels, x=pct_run, name="Between-Run (V_run)", orientation="h", marker_color="#7c3aed", text=[f"{v:.1f}%" if v >= 5 else "" for v in pct_run], textposition="inside"))
    fig.add_trace(go.Bar(y=levels, x=pct_day, name="Between-Day (V_day)", orientation="h", marker_color="#d97706", text=[f"{v:.1f}%" if v >= 5 else "" for v in pct_day], textposition="inside"))

    fig.update_layout(
        barmode="stack",
        title="Within-Laboratory Variance Decomposition (%)",
        xaxis_title="Share of Total Variance (%)",
        yaxis_title="QC Level",
        xaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=max(260, 90 * len(prec_results) + 130),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def build_fagan_nomogram(pre_prob: float, lr_pos: float, lr_neg: float) -> go.Figure:
    def prob_to_logit(p):
        p_c = np.clip(p, 1e-4, 1.0 - 1e-4)
        return np.log10(p_c / (1.0 - p_c))

    pre_odds = pre_prob / (1.0 - pre_prob)
    post_prob_pos = (pre_odds * lr_pos) / (1.0 + pre_odds * lr_pos)
    post_prob_neg = (pre_odds * lr_neg) / (1.0 + pre_odds * lr_neg)

    prob_ticks = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    lr_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0] * len(prob_ticks), y=[prob_to_logit(p) for p in prob_ticks], mode="markers+text", text=[f"{p*100:.1f}%" for p in prob_ticks], textposition="middle left", marker=dict(color="#475569", size=6), showlegend=False))
    fig.add_trace(go.Scatter(x=[1] * len(lr_ticks), y=[0.5 * np.log10(lr) for lr in lr_ticks], mode="markers+text", text=[f"{lr:g}" for lr in lr_ticks], textposition="top center", marker=dict(color="#0f172a", size=6), showlegend=False))
    fig.add_trace(go.Scatter(x=[2] * len(prob_ticks), y=[-prob_to_logit(p) for p in prob_ticks], mode="markers+text", text=[f"{p*100:.1f}%" for p in prob_ticks], textposition="middle right", marker=dict(color="#475569", size=6), showlegend=False))

    fig.add_trace(go.Scatter(
        x=[0, 1, 2],
        y=[prob_to_logit(pre_prob), 0.5 * np.log10(np.clip(lr_pos, 1e-3, 1e4)), -prob_to_logit(post_prob_pos)],
        mode="lines+markers",
        name=f"Positive Assay (LR+ = {lr_pos:.2f} → Post-Test = {post_prob_pos:.1%})",
        line=dict(color="#dc2626", width=3),
        marker=dict(size=10),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1, 2],
        y=[prob_to_logit(pre_prob), 0.5 * np.log10(np.clip(lr_neg, 1e-4, 1e3)), -prob_to_logit(post_prob_neg)],
        mode="lines+markers",
        name=f"Negative Assay (LR− = {lr_neg:.3f} → Post-Test = {post_prob_neg:.1%})",
        line=dict(color="#0284c7", width=3),
        marker=dict(size=10),
    ))

    fig.update_layout(
        title="Bayesian Fagan's Nomogram",
        xaxis=dict(tickmode="array", tickvals=[0, 1, 2], ticktext=["Pre-Test Probability", "Likelihood Ratio (LR)", "Post-Test Probability"], range=[-0.35, 2.35], showgrid=False),
        yaxis=dict(visible=False, range=[-3.2, 3.2]),
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


# ==============================================================================
# MAIN INTERFACE
# ==============================================================================

def main() -> None:
    st.title("🔬 ClinVal Biostats — Analytical Metrics & Matrices")

    with st.sidebar:
        st.header("1. Clinical Cohort Input")
        source_mode = st.radio(
            "Dataset Source",
            ["Built-in Demo: hs-cTnI Cohort (N=240)", "Upload Bench File (CSV / XLSX)"],
        )

        if source_mode == "Built-in Demo: hs-cTnI Cohort (N=240)":
            raw_df = generate_demo_dataset()
            raw_bytes = raw_df.to_csv(index=False).encode("utf-8")
        else:
            uploaded = st.file_uploader("Upload Raw Assay Data", type=["csv", "xlsx"])
            if uploaded is None:
                st.info("Upload a CSV or XLSX file to inspect metrics and matrices.")
                st.stop()
            raw_bytes = uploaded.getvalue()
            raw_df = pd.read_excel(io.BytesIO(raw_bytes)) if uploaded.name.lower().endswith(".xlsx") else pd.read_csv(io.BytesIO(raw_bytes))

        cols = list(raw_df.columns)

        def _guess_col(keywords: List[str], fallback_idx: int = 0) -> int:
            for i, c in enumerate(cols):
                if any(k in c.lower() for k in keywords):
                    return i
            return min(fallback_idx, len(cols) - 1)

        index_col = st.selectbox("Index Assay Column", cols, index=_guess_col(["index", "new", "test"], 0))
        ref_col = st.selectbox("Reference Standard Column", cols, index=_guess_col(["ref", "gold", "pred"], min(1, len(cols) - 1)))

        diag_options = ["None (Quantitative Only)"] + cols
        default_diag = _guess_col(["diag", "status", "outcome", "class", "target"], 0) + 1
        diag_choice = st.selectbox("Binary Diagnosis Column (0/1)", diag_options, index=default_diag)
        diag_col = None if diag_choice == "None (Quantitative Only)" else diag_choice

        comp_options = ["None"] + cols
        comp_guess = next((i + 1 for i, c in enumerate(cols) if "comp" in c.lower()), 0)
        comp_choice = st.selectbox("Comparator Assay (Optional)", comp_options, index=comp_guess)
        comp_col = None if comp_choice == "None" else comp_choice

        id_options = ["None"] + cols
        id_guess = next((i + 1 for i, c in enumerate(cols) if "id" in c.lower()), 0)
        id_choice = st.selectbox("Sample / Patient ID Column", id_options, index=id_guess)
        id_col = None if id_choice == "None" else id_choice

        st.divider()
        st.header("2. Model Parameters")
        exclude_outliers = st.checkbox("Exclude Rosner ESD Outliers", value=False)
        ba_mode = st.radio("Bland-Altman Scale", ["absolute", "percentage"], format_func=lambda m: "Absolute (Y − X)" if m == "absolute" else "Percentage (%)")
        error_ratio = st.number_input("Deming Variance Ratio λ", min_value=0.05, max_value=20.0, value=1.0, step=0.1)
        mdp_input = st.text_input("Medical Decision Points", value="14.0, 52.0")

    etl_rep = run_clinical_etl(
        df=raw_df,
        raw_bytes=raw_bytes,
        index_col=index_col,
        ref_col=ref_col,
        diag_col=diag_col,
        comp_col=comp_col,
        id_col=id_col,
    )

    if etl_rep.unit_warning:
        st.warning(etl_rep.unit_warning)

    analysis_df = (
        etl_rep.clean_df[~etl_rep.clean_df["_is_outlier"]].reset_index(drop=True)
        if exclude_outliers
        else etl_rep.clean_df.copy()
    )

    if len(analysis_df) < 10:
        st.error("Insufficient valid paired observations (< 10 rows) after ETL cleaning.")
        st.stop()

    x_vals = analysis_df["_ref_num"].to_numpy()
    y_vals = analysis_df["_index_num"].to_numpy()
    has_binary = diag_col is not None and "_diag_bin" in analysis_df.columns and analysis_df["_diag_bin"].nunique() == 2

    mdp_list: List[float] = []
    for token in mdp_input.split(","):
        token = token.strip()
        if token:
            try:
                mdp_list.append(float(token))
            except ValueError:
                pass

    pb_res = passing_bablok_regression(x_vals, y_vals)
    dem_res = deming_regression(x_vals, y_vals, error_ratio=error_ratio)
    ba_res = bland_altman_analysis(x_vals, y_vals, mode=ba_mode)
    mdp_df = estimate_mdp_bias(mdp_list, pb_res.slope, pb_res.slope_ci, pb_res.intercept, pb_res.intercept_ci)

    delong_res = None
    roc_df = None
    cutoffs = {}
    comp_delong_res = None
    comp_roc_df = None
    paired_res = None

    if has_binary:
        y_bin = analysis_df["_diag_bin"].to_numpy(dtype=int)
        delong_res = delong_roc_ci(y_bin, y_vals)
        roc_df, cutoffs = compute_roc_and_cutoffs(y_bin, y_vals)

        if comp_col and "_comp_num" in analysis_df.columns and analysis_df["_comp_num"].notna().all():
            comp_vals = analysis_df["_comp_num"].to_numpy()
            comp_delong_res = delong_roc_ci(y_bin, comp_vals)
            comp_roc_df, _ = compute_roc_and_cutoffs(y_bin, comp_vals)
            paired_res = paired_delong_test(y_bin, y_vals, comp_vals)

    # Top Summary Metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Validated Pairs (N)", f"{len(analysis_df)}", delta=f"{len(etl_rep.outlier_indices)} ESD outliers", delta_color="off")
    if delong_res:
        k2.metric("DeLong ROC-AUC", f"{delong_res.auc:.3f}", delta=f"95% CI: {delong_res.ci_lower:.3f}–{delong_res.ci_upper:.3f}", delta_color="off")
    else:
        k2.metric("Lin's Concordance (CCC)", f"{dem_res.lin_ccc:.4f}", delta=f"Pearson r: {dem_res.pearson_r:.4f}", delta_color="off")
    k3.metric("Passing-Bablok Slope", f"{pb_res.slope:.3f}", delta=f"95% CI: {pb_res.slope_ci[0]:.3f}–{pb_res.slope_ci[1]:.3f}", delta_color="off")
    k4.metric("Bland-Altman Bias", f"{ba_res.mean_bias:.2f}", delta=f"LoA: {ba_res.loa_lower:.2f} to {ba_res.loa_upper:.2f}", delta_color="off")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. ETL & Outlier Matrix",
        "2. ROC & 2×2 Contingency Matrix",
        "3. Method Comparison Matrices",
        "4. CLSI EP05 Precision & ANOVA",
        "5. Subgroups & Fagan Nomogram",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: ETL & OUTLIER MATRIX
    # --------------------------------------------------------------------------
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows", etl_rep.raw_rows)
        c2.metric("Valid Pairs", etl_rep.valid_paired_rows)
        c3.metric("Missing Dropped", etl_rep.missing_dropped)
        c4.metric("Censored Flags (<LoD / >AMR)", etl_rep.censored_parsed)

        if etl_rep.censored_records:
            st.markdown("#### Censored Value Substitution Log")
            st.dataframe(pd.DataFrame(etl_rep.censored_records), use_container_width=True)

        st.markdown("#### Rosner Generalized ESD Outlier Matrix")
        if etl_rep.outlier_table.empty:
            st.info("No statistical outliers detected at α = 0.05.")
        else:
            st.dataframe(etl_rep.outlier_table, use_container_width=True)

        with st.expander("Cleaned Analytical Data Matrix"):
            st.dataframe(analysis_df.drop(columns=[c for c in analysis_df.columns if c.startswith("_")]), use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 2: ROC & 2x2 CONTINGENCY MATRIX
    # --------------------------------------------------------------------------
    with tab2:
        if not has_binary:
            st.info("Map a binary Diagnosis column (0/1) in the sidebar to compute ROC and 2×2 contingency metrics.")
            eval_res = None
        else:
            col_ctrl1, col_ctrl2 = st.columns([1.3, 1.0])
            with col_ctrl1:
                cutoff_preset = st.selectbox("Decision Cutoff Strategy", list(cutoffs.keys()) + ["Custom Manual Threshold"])
            with col_ctrl2:
                if cutoff_preset == "Custom Manual Threshold":
                    active_threshold = st.number_input("Custom Cutoff Value", value=float(np.median(y_vals)), format="%.4f")
                else:
                    active_threshold = cutoffs[cutoff_preset]
                    st.metric("Active Threshold", f"{active_threshold:.4f}")

            target_prev = st.slider(
                "Target Clinical Prevalence (%) — Bayes PPV/NPV Adjustment",
                min_value=0.5, max_value=50.0, value=round(float(np.mean(y_bin) * 100.0), 1), step=0.5,
            ) / 100.0

            eval_res = evaluate_cutoff(y_bin, y_vals, active_threshold, prevalence=target_prev)

            col_roc, col_mat = st.columns([1.4, 1.0])
            with col_roc:
                st.plotly_chart(build_roc_figure(roc_df, delong_res, cutoffs, active_threshold, comp_roc_df, comp_delong_res), use_container_width=True, config=PLOTLY_CONFIG)

            with col_mat:
                st.markdown("#### 2×2 Contingency Matrix")
                cm_df = pd.DataFrame(
                    [
                        [eval_res["tp"], eval_res["fp"], eval_res["tp"] + eval_res["fp"]],
                        [eval_res["fn"], eval_res["tn"], eval_res["fn"] + eval_res["tn"]],
                        [eval_res["tp"] + eval_res["fn"], eval_res["fp"] + eval_res["tn"], len(y_bin)],
                    ],
                    index=["Index Positive (≥ Cutoff)", "Index Negative (< Cutoff)", "Total"],
                    columns=["Disease Positive (1)", "Disease Negative (0)", "Total"],
                )
                st.dataframe(cm_df, use_container_width=True)

                st.markdown("#### Diagnostic Accuracy Metrics")
                perf_table = pd.DataFrame([
                    {"Metric": "Sensitivity (Wilson 95% CI)", "Value": f"{eval_res['sensitivity']:.2%}", "95% CI": f"{eval_res['sens_wilson'][0]:.2%} – {eval_res['sens_wilson'][1]:.2%}"},
                    {"Metric": "Sensitivity (Clopper-Pearson Exact)", "Value": f"{eval_res['sensitivity']:.2%}", "95% CI": f"{eval_res['sens_exact'][0]:.2%} – {eval_res['sens_exact'][1]:.2%}"},
                    {"Metric": "Specificity (Wilson 95% CI)", "Value": f"{eval_res['specificity']:.2%}", "95% CI": f"{eval_res['spec_wilson'][0]:.2%} – {eval_res['spec_wilson'][1]:.2%}"},
                    {"Metric": "Specificity (Clopper-Pearson Exact)", "Value": f"{eval_res['specificity']:.2%}", "95% CI": f"{eval_res['spec_exact'][0]:.2%} – {eval_res['spec_exact'][1]:.2%}"},
                    {"Metric": "Positive Likelihood Ratio (LR+)", "Value": f"{eval_res['lr_pos']:.2f}", "95% CI": f"{eval_res['lr_pos_ci'][0]:.2f} – {eval_res['lr_pos_ci'][1]:.2f}"},
                    {"Metric": "Negative Likelihood Ratio (LR−)", "Value": f"{eval_res['lr_neg']:.3f}", "95% CI": f"{eval_res['lr_neg_ci'][0]:.3f} – {eval_res['lr_neg_ci'][1]:.3f}"},
                    {"Metric": "Diagnostic Odds Ratio (DOR)", "Value": f"{eval_res['dor']:.2f}", "95% CI": f"{eval_res['dor_ci'][0]:.2f} – {eval_res['dor_ci'][1]:.2f}"},
                    {"Metric": f"Bayes PPV ({target_prev:.1%} prev.)", "Value": f"{eval_res['ppv']:.2%}", "95% CI": "—"},
                    {"Metric": f"Bayes NPV ({target_prev:.1%} prev.)", "Value": f"{eval_res['npv']:.2%}", "95% CI": "—"},
                ])
                st.dataframe(perf_table, hide_index=True, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 3: METHOD COMPARISON MATRICES
    # --------------------------------------------------------------------------
    with tab3:
        c_reg, c_ba = st.columns(2)
        with c_reg:
            st.plotly_chart(build_method_comparison_figure(x_vals, y_vals, pb_res, dem_res, ref_col, index_col), use_container_width=True, config=PLOTLY_CONFIG)
        with c_ba:
            st.plotly_chart(build_bland_altman_figure(ba_res), use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("#### Regression & Agreement Parameter Matrix")
        summary_mc = pd.DataFrame([
            {
                "Model": "Passing-Bablok",
                "Slope / Bias (95% CI)": f"{pb_res.slope:.4f} ({pb_res.slope_ci[0]:.4f} to {pb_res.slope_ci[1]:.4f})",
                "Intercept / LoA (95% CI)": f"{pb_res.intercept:.4f} ({pb_res.intercept_ci[0]:.4f} to {pb_res.intercept_ci[1]:.4f})",
                "Diagnostic Metric": f"CUSUM H = {pb_res.cusum_h:.3f} (Crit: {pb_res.cusum_critical})",
            },
            {
                "Model": f"Deming (λ = {dem_res.error_ratio:.2f})",
                "Slope / Bias (95% CI)": f"{dem_res.slope:.4f} ({dem_res.slope_ci[0]:.4f} to {dem_res.slope_ci[1]:.4f})",
                "Intercept / LoA (95% CI)": f"{dem_res.intercept:.4f} ({dem_res.intercept_ci[0]:.4f} to {dem_res.intercept_ci[1]:.4f})",
                "Diagnostic Metric": f"Lin's CCC = {dem_res.lin_ccc:.4f} | r = {dem_res.pearson_r:.4f}",
            },
            {
                "Model": f"Bland-Altman ({ba_res.mode.title()})",
                "Slope / Bias (95% CI)": f"{ba_res.mean_bias:.3f} ({ba_res.mean_bias_ci[0]:.3f} to {ba_res.mean_bias_ci[1]:.3f})",
                "Intercept / LoA (95% CI)": f"{ba_res.loa_lower:.3f} to {ba_res.loa_upper:.3f}",
                "Diagnostic Metric": f"Prop. Slope = {ba_res.prop_slope:.4f} (p = {ba_res.prop_p_value:.4f})",
            },
        ])
        st.dataframe(summary_mc, hide_index=True, use_container_width=True)

        if not mdp_df.empty:
            st.markdown("#### Medical Decision Point (MDP) Bias Matrix")
            st.dataframe(mdp_df, hide_index=True, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 4: STANDALONE CLSI EP05 PRECISION & VARIANCE COMPONENTS
    # --------------------------------------------------------------------------
    with tab4:
        prec_source = st.radio(
            "CLSI EP05 Precision Input",
            [
                "Built-in Demo: CLSI EP05-A3 20×2×2 Multi-Level QC Cohort",
                "Upload Custom Multi-Day QC Replicates (CSV / XLSX)",
            ],
            horizontal=True,
        )

        prec_df_raw: Optional[pd.DataFrame] = None
        if prec_source.startswith("Built-in Demo"):
            prec_df_raw = generate_ep05_demo_dataset()
        else:
            up_prec = st.file_uploader("Upload Multi-Day QC File", type=["csv", "xlsx"], key="ep05_uploader")
            if up_prec is not None:
                p_bytes = up_prec.getvalue()
                prec_df_raw = pd.read_excel(io.BytesIO(p_bytes)) if up_prec.name.lower().endswith(".xlsx") else pd.read_csv(io.BytesIO(p_bytes))

        if prec_df_raw is None:
            st.info("Upload a multi-day QC dataset (Columns: QC_Level, Day, Run, Measured_Value) to inspect variance components.")
        else:
            p_cols = list(prec_df_raw.columns)

            def _guess_pcol(keywords: List[str], fallback: int = 0) -> int:
                for i, c in enumerate(p_cols):
                    if any(k in c.lower() for k in keywords):
                        return i
                return min(fallback, len(p_cols) - 1)

            pc1, pc2, pc3, pc4 = st.columns(4)
            with pc1:
                p_lvl_options = ["None (Single QC Pool)"] + p_cols
                p_lvl_guess = next((i + 1 for i, c in enumerate(p_cols) if any(k in c.lower() for k in ["level", "pool", "qc", "sample"])), 0)
                p_lvl_choice = st.selectbox("QC Level Column", p_lvl_options, index=p_lvl_guess)
                p_lvl_col = None if p_lvl_choice == "None (Single QC Pool)" else p_lvl_choice
            with pc2:
                p_day_col = st.selectbox("Day / Batch Column", p_cols, index=_guess_pcol(["day", "batch", "date"], 0))
            with pc3:
                p_run_options = ["None (1-Factor EP15)"] + p_cols
                p_run_guess = next((i + 1 for i, c in enumerate(p_cols) if "run" in c.lower()), 0)
                p_run_choice = st.selectbox("Run Column (2-Factor EP05)", p_run_options, index=p_run_guess)
                p_run_col = None if p_run_choice.startswith("None") else p_run_choice
            with pc4:
                p_val_col = st.selectbox("Measured Value Column", p_cols, index=_guess_pcol(["val", "meas", "result", "conc"], len(p_cols) - 1))

            claim_r_col = next((c for c in p_cols if "claim" in c.lower() and "cvr" in c.lower()), None)
            claim_wl_col = next((c for c in p_cols if "claim" in c.lower() and "cvwl" in c.lower()), None)

            try:
                prec_results = compute_clsi_ep05_precision(
                    df=prec_df_raw,
                    value_col=p_val_col,
                    day_col=p_day_col,
                    qc_level_col=p_lvl_col,
                    run_col=p_run_col,
                    claim_cvr_col=claim_r_col,
                    claim_cvwl_col=claim_wl_col,
                )

                st.markdown(f"**Active ANOVA Design:** `{prec_results[0].model_design}`")
                st.dataframe(format_precision_summary_table(prec_results), hide_index=True, use_container_width=True)
                st.plotly_chart(build_variance_components_figure(prec_results), use_container_width=True, config=PLOTLY_CONFIG)

                selected_lvl = st.selectbox("Select QC Level for Nested ANOVA Matrix & Trajectory:", [r.qc_level for r in prec_results])
                active_prec = next(r for r in prec_results if r.qc_level == selected_lvl)

                v_col1, v_col2 = st.columns([1.4, 1.0])
                with v_col1:
                    st.plotly_chart(build_precision_trajectory_figure(active_prec), use_container_width=True, config=PLOTLY_CONFIG)
                with v_col2:
                    st.markdown(f"#### Nested ANOVA Matrix ({active_prec.qc_level})")
                    st.dataframe(active_prec.anova_table, hide_index=True, use_container_width=True)
                    st.markdown(
                        f"- **Satterthwaite df ($\nu_{{WL}}$):** `{active_prec.df_wl:.2f}`\n"
                        f"- **Repeatability df ($\nu_r$):** `{int(active_prec.df_r)}`\n"
                        f"- **Between-Run SD ($S_{{run}}$):** `{active_prec.s_run:.4f}` (`{active_prec.cv_run:.2f}%`)\n"
                        f"- **Between-Day SD ($S_{{day}}$):** `{active_prec.s_day:.4f}` (`{active_prec.cv_day:.2f}%`)"
                    )
            except ValueError as err:
                st.error(f"EP05 Input Error: {err}")

    # --------------------------------------------------------------------------
    # TAB 5: SUBGROUPS & FAGAN NOMOGRAM
    # --------------------------------------------------------------------------
    with tab5:
        if paired_res is not None:
            st.markdown("#### Paired DeLong AUC Comparison Matrix")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Index AUC", f"{paired_res.auc_1:.4f}", f"{paired_res.ci_1[0]:.3f}–{paired_res.ci_1[1]:.3f}", delta_color="off")
            p2.metric("Comparator AUC", f"{paired_res.auc_2:.4f}", f"{paired_res.ci_2[0]:.3f}–{paired_res.ci_2[1]:.3f}", delta_color="off")
            p3.metric("ΔAUC", f"{paired_res.auc_diff:+.4f}", f"SE: {paired_res.se_diff:.4f}", delta_color="off")
            p4.metric("DeLong p-value", f"{paired_res.p_value:.4g}", delta_color="off")

        excluded_meta = {index_col, ref_col, diag_col, comp_col, id_col}
        cat_candidates = [c for c in analysis_df.columns if not c.startswith("_") and c not in excluded_meta and analysis_df[c].nunique() <= 8]
        if cat_candidates:
            st.markdown("#### Subgroup Stratification Matrix")
            strat_col = st.selectbox("Stratification Variable", cat_candidates)
            sub_rows = []
            for grp_val, grp_df in analysis_df.groupby(strat_col, dropna=False):
                if len(grp_df) < 8:
                    continue
                gx = grp_df["_ref_num"].to_numpy()
                gy = grp_df["_index_num"].to_numpy()
                g_ba = bland_altman_analysis(gx, gy, mode="absolute")
                row_info = {
                    "Subgroup": str(grp_val),
                    "N": len(grp_df),
                    "Mean Bias": round(g_ba.mean_bias, 3),
                    "95% LoA": f"{g_ba.loa_lower:.2f} to {g_ba.loa_upper:.2f}",
                }
                if has_binary and grp_df["_diag_bin"].nunique() == 2 and grp_df["_diag_bin"].sum() >= 2:
                    g_dl = delong_roc_ci(grp_df["_diag_bin"].to_numpy(dtype=int), gy)
                    row_info["DeLong AUC (95% CI)"] = f"{g_dl.auc:.3f} ({g_dl.ci_lower:.3f}–{g_dl.ci_upper:.3f})"
                sub_rows.append(row_info)
            if sub_rows:
                st.dataframe(pd.DataFrame(sub_rows), hide_index=True, use_container_width=True)

        if has_binary and eval_res is not None:
            st.plotly_chart(build_fagan_nomogram(eval_res["target_prevalence"], eval_res["lr_pos"], eval_res["lr_neg"]), use_container_width=True, config=PLOTLY_CONFIG)


if __name__ == "__main__":
    main()
