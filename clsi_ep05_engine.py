"""
clsi_ep05_engine.py
Standalone CLSI EP05-A3 & EP15-A3 Precision and Nested Variance Components Engine.

Zero dependencies on stats_engine.py or UI frameworks.
Can be imported and verified independently in unit tests or external pipelines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PrecisionLevelResult:
    qc_level: str
    model_design: str
    n_total: int
    n_days: int
    n_runs: int
    mean: float
    sd_overall: float
    cv_overall: float
    # Within-Run Repeatability
    s_r: float
    cv_r: float
    cv_r_ci: Tuple[float, float]
    df_r: float
    # Intermediate Components (Between-Run & Between-Day)
    s_run: float
    cv_run: float
    s_day: float
    cv_day: float
    # Within-Laboratory Total Imprecision
    s_wl: float
    cv_wl: float
    cv_wl_ci: Tuple[float, float]
    df_wl: float
    # Variance Breakdown (%)
    pct_var_r: float
    pct_var_run: float
    pct_var_day: float
    # Claim & Upper Verification Limit (UVL)
    claimed_cv_r: Optional[float]
    uvl_cv_r: Optional[float]
    status_cv_r: str
    claimed_cv_wl: Optional[float]
    uvl_cv_wl: Optional[float]
    status_cv_wl: str
    anova_table: pd.DataFrame
    clean_records: pd.DataFrame


def _parse_numeric_series(series: pd.Series) -> pd.Series:
    """Coerces numeric lab values and semi-quantitative flags (e.g. '<1.5') to float."""
    out = np.full(len(series), np.nan, dtype=float)
    for idx, val in enumerate(series):
        if pd.isna(val):
            continue
        if isinstance(val, (int, float, np.number)):
            out[idx] = float(val)
            continue
        s = str(val).strip()
        match = re.match(r"^([<>]=?|≤|≥)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)$", s)
        if match:
            op, num_str = match.groups()
            num = float(num_str)
            out[idx] = num * 0.5 if op in ("<", "<=", "≤") else num
        else:
            try:
                out[idx] = float(s)
            except ValueError:
                out[idx] = np.nan
    return pd.Series(out, index=series.index)


def _chi2_sd_ci(sd_val: float, df: float, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact Chi-Square confidence interval for SD with Satterthwaite degrees of freedom."""
    if sd_val <= 0 or df <= 0:
        return (0.0, 0.0)
    chi2_low = stats.chi2.ppf(alpha / 2.0, df=df)
    chi2_high = stats.chi2.ppf(1.0 - alpha / 2.0, df=df)
    sd_lower = sd_val * np.sqrt(df / chi2_high)
    sd_upper = sd_val * np.sqrt(df / chi2_low)
    return (float(sd_lower), float(sd_upper))


def _evaluate_uvl(
    obs_cv: float,
    claim_cv: Optional[float],
    df: float,
    alpha_uvl: float = 0.05,
) -> Tuple[Optional[float], str]:
    """Computes Upper Verification Limit: UVL = Claim * sqrt(chi2_{1 - alpha, df} / df)."""
    if claim_cv is None or claim_cv <= 0 or df <= 0:
        return None, "No Claim Specified"
    f_uvl = np.sqrt(stats.chi2.ppf(1.0 - alpha_uvl, df=df) / df)
    uvl = float(claim_cv * f_uvl)
    if obs_cv <= claim_cv:
        return uvl, "PASS (≤ Claim)"
    if obs_cv <= uvl:
        return uvl, "PASS (≤ UVL)"
    return uvl, "FAIL (> UVL)"


def generate_ep05_demo_dataset(seed: int = 108) -> pd.DataFrame:
    """
    Generates a standard CLSI EP05-A3 20-day x 2-run x 2-replicate (N=80 per level)
    multi-level QC precision dataset across 3 clinical concentrations.
    """
    rng = np.random.default_rng(seed)
    levels = [
        {"name": "QC-1 (Low)", "target": 12.5, "sd_r": 0.48, "sd_run": 0.32, "sd_day": 0.42, "claim_cvr": 4.5, "claim_cvwl": 6.5},
        {"name": "QC-2 (Mid / MDP)", "target": 48.0, "sd_r": 1.45, "sd_run": 0.90, "sd_day": 1.25, "claim_cvr": 3.5, "claim_cvwl": 5.0},
        {"name": "QC-3 (High)", "target": 215.0, "sd_r": 5.10, "sd_run": 3.20, "sd_day": 4.10, "claim_cvr": 3.0, "claim_cvwl": 4.2},
    ]

    records = []
    for lvl in levels:
        for d in range(1, 21):
            day_eff = rng.normal(0.0, lvl["sd_day"])
            for run_label in ("Run-1 (AM)", "Run-2 (PM)"):
                run_eff = rng.normal(0.0, lvl["sd_run"])
                for rep in (1, 2):
                    rep_err = rng.normal(0.0, lvl["sd_r"])
                    val = max(0.1, lvl["target"] + day_eff + run_eff + rep_err)
                    records.append({
                        "QC_Level": lvl["name"],
                        "Day": f"Day {d:02d}",
                        "Run": run_label,
                        "Replicate": rep,
                        "Measured_Value": round(val, 3),
                        "Claimed_CVr_Pct": lvl["claim_cvr"],
                        "Claimed_CVwl_Pct": lvl["claim_cvwl"],
                    })

    return pd.DataFrame(records)


def compute_clsi_ep05_precision(
    df: pd.DataFrame,
    value_col: str,
    day_col: str,
    qc_level_col: Optional[str] = None,
    run_col: Optional[str] = None,
    claim_cvr_col: Optional[str] = None,
    claim_cvwl_col: Optional[str] = None,
    manual_claims: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    bonferroni_multilevel: bool = False,
    alpha: float = 0.05,
) -> List[PrecisionLevelResult]:
    """
    Computes CLSI EP05-A3 (2-Factor Nested: Day > Run > Replicate) or CLSI EP15-A3
    (1-Factor Nested: Day > Replicate) random-effects ANOVA variance components.
    """
    work_df = df.copy()
    work_df["_val"] = _parse_numeric_series(work_df[value_col])
    work_df = work_df.dropna(subset=["_val", day_col]).copy()

    if qc_level_col and qc_level_col in work_df.columns:
        work_df["_level"] = work_df[qc_level_col].astype(str)
    else:
        work_df["_level"] = "QC Pool 1"

    levels = list(work_df["_level"].unique())
    n_levels = max(1, len(levels))
    alpha_uvl = (alpha / n_levels) if bonferroni_multilevel else alpha

    results: List[PrecisionLevelResult] = []

    for lvl in levels:
        sub = work_df[work_df["_level"] == lvl].copy().reset_index(drop=True)
        y = sub["_val"].to_numpy(dtype=float)
        n_total = len(y)
        grand_mean = float(np.mean(y))
        sd_overall = float(np.std(y, ddof=1)) if n_total > 1 else 0.0
        cv_overall = (100.0 * sd_overall / abs(grand_mean)) if abs(grand_mean) > 1e-12 else 0.0

        days = sub[day_col].astype(str)
        d_count = days.nunique()
        if d_count < 2:
            raise ValueError(f"QC Level '{lvl}' requires at least 2 distinct days/batches (found {d_count}).")

        use_two_factor = False
        if run_col and run_col in sub.columns:
            sub["_day_run"] = days + "||" + sub[run_col].astype(str)
            m_runs = sub["_day_run"].nunique()
            if m_runs > d_count and n_total > m_runs:
                use_two_factor = True

        if not use_two_factor:
            # 1-Factor Nested Random-Effects ANOVA (Day > Replicate)
            model_design = "1-Factor Nested ANOVA (Day > Replicate | CLSI EP15-A3)"
            grp_stats = sub.groupby(days, sort=False)["_val"].agg(["count", "mean"])
            n_i = grp_stats["count"].to_numpy(dtype=float)
            y_bar_i = grp_stats["mean"].to_numpy(dtype=float)

            ss_b = float(np.sum(n_i * (y_bar_i - grand_mean) ** 2))
            day_mean_map = dict(zip(grp_stats.index, grp_stats["mean"]))
            fitted = days.map(day_mean_map).to_numpy(dtype=float)
            ss_w = float(np.sum((y - fitted) ** 2))

            df_b = float(d_count - 1)
            df_w = float(n_total - d_count)
            if df_w <= 0:
                raise ValueError(f"QC Level '{lvl}' requires replicate measurements within days (N > Days).")

            ms_b = ss_b / df_b
            ms_w = ss_w / df_w
            n_0 = (n_total - np.sum(n_i ** 2) / n_total) / df_b

            v_r = ms_w
            v_day = max(0.0, (ms_b - ms_w) / n_0)
            v_run = 0.0
            v_wl = v_r + v_day

            if v_day > 0:
                c_b = 1.0 / n_0
                c_w = 1.0 - c_b
                num = (c_b * ms_b + c_w * ms_w) ** 2
                den = ((c_b * ms_b) ** 2) / df_b + ((c_w * ms_w) ** 2) / df_w
                df_wl = float(np.clip(num / den, df_b, n_total - 1))
            else:
                df_wl = float(df_w)

            n_runs_out = d_count
            anova_df = pd.DataFrame([
                {"Source of Variation": "Between-Day", "SS": round(ss_b, 5), "df": int(df_b), "MS": round(ms_b, 5), "Variance Component": round(v_day, 5)},
                {"Source of Variation": "Within-Run (Repeatability)", "SS": round(ss_w, 5), "df": int(df_w), "MS": round(ms_w, 5), "Variance Component": round(v_r, 5)},
                {"Source of Variation": "Total (Within-Laboratory)", "SS": round(ss_b + ss_w, 5), "df": int(df_b + df_w), "MS": np.nan, "Variance Component": round(v_wl, 5)},
            ])
        else:
            # 2-Factor Nested Random-Effects ANOVA (Day > Run > Replicate | CLSI EP05-A3)
            model_design = "2-Factor Nested ANOVA (Day > Run > Replicate | CLSI EP05-A3)"
            day_grp = sub.groupby(days, sort=False)["_val"].agg(["count", "mean"])
            run_grp = sub.groupby("_day_run", sort=False)["_val"].agg(["count", "mean"])

            n_i = day_grp["count"].to_numpy(dtype=float)
            y_bar_i = day_grp["mean"].to_numpy(dtype=float)
            n_ij = run_grp["count"].to_numpy(dtype=float)
            y_bar_ij = run_grp["mean"].to_numpy(dtype=float)

            m_runs = len(run_grp)
            run_days = [k.split("||")[0] for k in run_grp.index]
            day_mean_map = dict(zip(day_grp.index, day_grp["mean"]))
            y_bar_i_for_run = np.array([day_mean_map[d_name] for d_name in run_days], dtype=float)

            ss_day = float(np.sum(n_i * (y_bar_i - grand_mean) ** 2))
            ss_run = float(np.sum(n_ij * (y_bar_ij - y_bar_i_for_run) ** 2))
            run_mean_map = dict(zip(run_grp.index, run_grp["mean"]))
            fitted_run = sub["_day_run"].map(run_mean_map).to_numpy(dtype=float)
            ss_w = float(np.sum((y - fitted_run) ** 2))

            df_day = float(d_count - 1)
            df_run = float(m_runs - d_count)
            df_w = float(n_total - m_runs)

            ms_day = ss_day / df_day
            ms_run = ss_run / df_run
            ms_w = ss_w / df_w

            # Henderson Method I unbalanced nested ANOVA coefficients
            sum_nij2_by_day = 0.0
            for _, d_sub in sub.groupby(days, sort=False):
                counts_in_day = d_sub.groupby("_day_run")["_val"].count().to_numpy(dtype=float)
                sum_nij2_by_day += np.sum(counts_in_day ** 2) / np.sum(counts_in_day)

            k1 = (n_total - sum_nij2_by_day) / df_run
            k2 = (sum_nij2_by_day - np.sum(n_ij ** 2) / n_total) / df_day
            k3 = (n_total - np.sum(n_i ** 2) / n_total) / df_day

            v_r = ms_w
            v_run = max(0.0, (ms_run - ms_w) / k1)
            v_day = max(0.0, (ms_day - ms_w - k2 * v_run) / k3)
            v_wl = v_r + v_run + v_day

            # Exact Satterthwaite linear combination weights matching clamped variance components
            if v_day > 0 and v_run > 0:
                c_day = 1.0 / k3
                c_run = (1.0 / k1) * (1.0 - (k2 / k3))
                c_w = 1.0 - c_day - c_run
            elif v_day > 0 and v_run == 0:
                c_day = 1.0 / k3
                c_run = 0.0
                c_w = 1.0 - c_day
            elif v_day == 0 and v_run > 0:
                c_day = 0.0
                c_run = 1.0 / k1
                c_w = 1.0 - c_run
            else:
                c_day, c_run, c_w = 0.0, 0.0, 1.0

            num = (c_day * ms_day + c_run * ms_run + c_w * ms_w) ** 2
            den = (
                (((c_day * ms_day) ** 2) / df_day if c_day > 0 else 0.0)
                + (((c_run * ms_run) ** 2) / df_run if c_run > 0 else 0.0)
                + (((c_w * ms_w) ** 2) / df_w if c_w > 0 else 0.0)
            )
            df_wl = float(np.clip(num / den, min(df_day, df_w), n_total - 1)) if den > 0 else float(df_w)
            n_runs_out = m_runs

            anova_df = pd.DataFrame([
                {"Source of Variation": "Between-Day", "SS": round(ss_day, 5), "df": int(df_day), "MS": round(ms_day, 5), "Variance Component": round(v_day, 5)},
                {"Source of Variation": "Between-Run (Within-Day)", "SS": round(ss_run, 5), "df": int(df_run), "MS": round(ms_run, 5), "Variance Component": round(v_run, 5)},
                {"Source of Variation": "Within-Run (Repeatability)", "SS": round(ss_w, 5), "df": int(df_w), "MS": round(ms_w, 5), "Variance Component": round(v_r, 5)},
                {"Source of Variation": "Total (Within-Laboratory)", "SS": round(ss_day + ss_run + ss_w, 5), "df": int(df_day + df_run + df_w), "MS": np.nan, "Variance Component": round(v_wl, 5)},
            ])

        scale = 100.0 / abs(grand_mean) if abs(grand_mean) > 1e-12 else 0.0

        s_r = float(np.sqrt(v_r))
        cv_r = s_r * scale
        sr_low, sr_high = _chi2_sd_ci(s_r, df_w, alpha)
        cv_r_ci = (sr_low * scale, sr_high * scale)

        s_run = float(np.sqrt(v_run))
        cv_run = s_run * scale

        s_day = float(np.sqrt(v_day))
        cv_day = s_day * scale

        s_wl = float(np.sqrt(v_wl))
        cv_wl = s_wl * scale
        swl_low, swl_high = _chi2_sd_ci(s_wl, df_wl, alpha)
        cv_wl_ci = (swl_low * scale, swl_high * scale)

        pct_var_r = (100.0 * v_r / v_wl) if v_wl > 0 else 100.0
        pct_var_run = (100.0 * v_run / v_wl) if v_wl > 0 else 0.0
        pct_var_day = (100.0 * v_day / v_wl) if v_wl > 0 else 0.0

        claim_r: Optional[float] = None
        claim_wl: Optional[float] = None
        if manual_claims and lvl in manual_claims:
            claim_r, claim_wl = manual_claims[lvl]
        else:
            if claim_cvr_col and claim_cvr_col in sub.columns and sub[claim_cvr_col].notna().any():
                claim_r = float(pd.to_numeric(sub[claim_cvr_col], errors="coerce").dropna().iloc[0])
            if claim_cvwl_col and claim_cvwl_col in sub.columns and sub[claim_cvwl_col].notna().any():
                claim_wl = float(pd.to_numeric(sub[claim_cvwl_col], errors="coerce").dropna().iloc[0])

        uvl_r, status_r = _evaluate_uvl(cv_r, claim_r, df_w, alpha_uvl)
        uvl_wl, status_wl = _evaluate_uvl(cv_wl, claim_wl, df_wl, alpha_uvl)

        sub_clean = pd.DataFrame({
            "QC_Level": lvl,
            "Day": days.to_numpy(),
            "Run": sub[run_col].astype(str).to_numpy() if (run_col and run_col in sub.columns) else "Run-1",
            "Value": y,
        })

        results.append(
            PrecisionLevelResult(
                qc_level=lvl,
                model_design=model_design,
                n_total=n_total,
                n_days=d_count,
                n_runs=n_runs_out,
                mean=grand_mean,
                sd_overall=sd_overall,
                cv_overall=cv_overall,
                s_r=s_r,
                cv_r=cv_r,
                cv_r_ci=cv_r_ci,
                df_r=df_w,
                s_run=s_run,
                cv_run=cv_run,
                s_day=s_day,
                cv_day=cv_day,
                s_wl=s_wl,
                cv_wl=cv_wl,
                cv_wl_ci=cv_wl_ci,
                df_wl=df_wl,
                pct_var_r=pct_var_r,
                pct_var_run=pct_var_run,
                pct_var_day=pct_var_day,
                claimed_cv_r=claim_r,
                uvl_cv_r=uvl_r,
                status_cv_r=status_r,
                claimed_cv_wl=claim_wl,
                uvl_cv_wl=uvl_wl,
                status_cv_wl=status_wl,
                anova_table=anova_df,
                clean_records=sub_clean,
            )
        )

    return results


def format_precision_summary_table(prec_results: List[PrecisionLevelResult]) -> pd.DataFrame:
    """Formats variance components into a structured precision summary matrix."""
    rows = []
    for r in prec_results:
        rows.append({
            "QC Level": r.qc_level,
            "N (Days / Runs)": f"{r.n_total} ({r.n_days}d / {r.n_runs}r)",
            "Mean": round(r.mean, 3),
            "Repeatability S_r": round(r.s_r, 3),
            "Repeatability CV_r (95% CI)": f"{r.cv_r:.2f}% ({r.cv_r_ci[0]:.2f}–{r.cv_r_ci[1]:.2f}%)",
            "Between-Run CV_run": f"{r.cv_run:.2f}%",
            "Between-Day CV_day": f"{r.cv_day:.2f}%",
            "Within-Lab S_WL": round(r.s_wl, 3),
            "Within-Lab CV_WL (95% CI)": f"{r.cv_wl:.2f}% ({r.cv_wl_ci[0]:.2f}–{r.cv_wl_ci[1]:.2f}%)",
            "UVL Check (CV_r / CV_WL)": f"{r.status_cv_r} | {r.status_cv_wl}",
        })
    return pd.DataFrame(rows)
