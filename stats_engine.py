"""
stats_engine.py
Deterministic Clinical Biostatistics & Diagnostic Assay Validation Engine.

References:
- DeLong ER, DeLong DM, Clarke-Pearson DL. Biometrics. 1988;44(3):837-845.
- Sun X, Xu W. IEEE Signal Processing Letters. 2014;21(11):1389-1393.
- Passing H, Bablok W. J Clin Chem Clin Biochem. 1983;21(11):709-720.
- Bland JM, Altman DG. Stat Methods Med Res. 1999;8(2):135-160.
- Rosner B. Technometrics. 1983;25(2):165-172 (CLSI EP09 outlier audit).
- Simel DL, Samsa GP, Matchar DB. J Clin Epidemiol. 1991;44(8):763-770.
"""

from __future__ import annotations

import hashlib
import platform
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn import metrics as skm


# ==============================================================================
# 1. SYNTHETIC CLINICAL BENCHMARK GENERATOR
# ==============================================================================

def generate_demo_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic clinical laboratory validation dataset (N = 240)
    comparing a novel High-Sensitivity Cardiac Troponin I (hs-cTnI) Index Assay
    and a secondary Comparator Biomarker against an adjudicated Reference Standard.
    Includes intentional bench edge cases (censored '<LoD' strings, missing values,
    and analytical outliers) to demonstrate automated ETL and Rosner ESD auditing.
    """
    rng = np.random.default_rng(seed)
    n_neg = 155
    n_pos = 85
    n_total = n_neg + n_pos

    # Non-ischemic chest pain cohort (log-normal baseline distribution, ng/L)
    ref_neg = rng.lognormal(mean=1.75, sigma=0.65, size=n_neg)
    # Acute Myocardial Infarction (AMI) cohort (elevated troponin distribution, ng/L)
    ref_pos = rng.lognormal(mean=4.45, sigma=0.85, size=n_pos)

    ref_vals = np.concatenate([ref_neg, ref_pos])
    diagnosis = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])

    # Index assay: slight proportional bias (+3.5%), small constant offset (+0.8 ng/L), heteroscedastic CV ~6%
    noise_idx = rng.normal(loc=0.0, scale=0.06 * ref_vals + 0.55)
    index_vals = np.maximum(0.5, 1.035 * ref_vals + 0.8 + noise_idx)

    # Comparator assay (slightly lower discrimination for paired DeLong testing)
    comp_neg = rng.lognormal(mean=1.95, sigma=0.80, size=n_neg)
    comp_pos = rng.lognormal(mean=4.15, sigma=0.95, size=n_pos)
    comp_vals = np.concatenate([comp_neg, comp_pos])

    # Metadata & pre-analytical covariates
    patient_ids = [f"PT-{1000 + i}" for i in range(n_total)]
    ages = np.concatenate([
        rng.integers(32, 78, size=n_neg),
        rng.integers(45, 88, size=n_pos)
    ])
    sexes = rng.choice(["Female", "Male"], size=n_total, p=[0.44, 0.56])
    matrices = rng.choice(["Li-Heparin Plasma", "Serum"], size=n_total, p=[0.65, 0.35])
    lots = rng.choice(["LOT-2026A", "LOT-2026B"], size=n_total, p=[0.52, 0.48])

    df = pd.DataFrame({
        "Patient_ID": patient_ids,
        "Diagnosis": diagnosis,
        "Reference_hs_cTnI_ngL": np.round(ref_vals, 2),
        "Index_hs_cTnI_ngL": np.round(index_vals, 2).astype(object),
        "Comparator_Assay_ngL": np.round(comp_vals, 2),
        "Age": ages,
        "Sex": sexes,
        "Sample_Matrix": matrices,
        "Reagent_Lot": lots,
    })

    # Shuffle cohort order
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Inject realistic clinical laboratory edge cases for ETL demonstration
    df.loc[4, "Index_hs_cTnI_ngL"] = "<1.50"       # Below Limit of Detection string
    df.loc[19, "Index_hs_cTnI_ngL"] = ">2500.0"    # Above Analytical Measuring Range string
    df.loc[42, "Index_hs_cTnI_ngL"] = np.nan      # QNS (Quantity Not Sufficient) missing value
    df.loc[88, "Index_hs_cTnI_ngL"] = float(df.loc[88, "Reference_hs_cTnI_ngL"]) * 3.4  # Heterophile antibody outlier
    df.loc[173, "Index_hs_cTnI_ngL"] = float(df.loc[173, "Reference_hs_cTnI_ngL"]) * 0.22  # Clot/micro-bubble outlier

    return df


# ==============================================================================
# 2. CLINICAL ETL, UNIT AUDIT & ROSNER ESD OUTLIER DETECTION
# ==============================================================================

@dataclass
class ETLReport:
    raw_rows: int
    valid_paired_rows: int
    missing_dropped: int
    censored_parsed: int
    censored_records: List[Dict[str, Any]]
    unit_warning: Optional[str]
    median_ratio: float
    outlier_indices: List[int]
    outlier_table: pd.DataFrame
    clean_df: pd.DataFrame
    dataset_sha256: str


def _parse_clinical_numeric(series: pd.Series) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    Parses numeric lab values while capturing semi-quantitative censored flags
    such as '<0.05' or '>1000' instead of silently failing.
    """
    parsed_vals = np.full(len(series), np.nan, dtype=float)
    censored_log: List[Dict[str, Any]] = []

    for idx, val in enumerate(series):
        if pd.isna(val):
            continue
        if isinstance(val, (int, float, np.number)):
            parsed_vals[idx] = float(val)
            continue

        s = str(val).strip()
        match = re.match(r"^([<>]=?|≤|≥)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)$", s)
        if match:
            op, num_str = match.groups()
            num = float(num_str)
            # Standard clinical convention: substitute LoD/2 for '<' and LoQ for '>'
            substituted = num * 0.5 if op in ("<", "<=", "≤") else num
            parsed_vals[idx] = substituted
            censored_log.append({
                "Row_Index": idx,
                "Raw_String": s,
                "Substituted_Value": substituted,
                "Rule": "LoD/2 substitution" if op in ("<", "<=", "≤") else "Upper AMR cap"
            })
        else:
            try:
                parsed_vals[idx] = float(s)
            except ValueError:
                parsed_vals[idx] = np.nan

    return pd.Series(parsed_vals, index=series.index), censored_log


def rosner_esd_test(
    differences: np.ndarray,
    max_outliers: Optional[int] = None,
    alpha: float = 0.05
) -> List[int]:
    """
    Rosner's Generalized Extreme Studentized Deviate (ESD) test for multiple outliers
    on method comparison differences (CLSI EP09c standard).
    Returns 0-based positional indices of confirmed outliers in `differences`.
    """
    n = len(differences)
    if n < 15:
        return []

    if max_outliers is None:
        max_outliers = max(1, min(10, int(np.floor(0.05 * n))))

    work_vals = differences.astype(float).copy()
    orig_indices = np.arange(n)
    candidate_indices: List[int] = []
    r_stats: List[float] = []
    lambdas: List[float] = []

    for i in range(max_outliers):
        curr_n = len(work_vals)
        if curr_n <= 3:
            break
        mean_v = np.mean(work_vals)
        sd_v = np.std(work_vals, ddof=1)
        if sd_v == 0:
            break

        abs_dev = np.abs(work_vals - mean_v)
        max_pos = int(np.argmax(abs_dev))
        r_i = abs_dev[max_pos] / sd_v

        # Critical value lambda_i
        p = 1.0 - alpha / (2.0 * (curr_n - 1 + 1))
        t_crit = stats.t.ppf(p, df=curr_n - 2)
        lambda_i = ((curr_n - 1) * t_crit) / np.sqrt((curr_n - 2 + t_crit ** 2) * curr_n)

        candidate_indices.append(int(orig_indices[max_pos]))
        r_stats.append(float(r_i))
        lambdas.append(float(lambda_i))

        work_vals = np.delete(work_vals, max_pos)
        orig_indices = np.delete(orig_indices, max_pos)

    # Largest k such that R_k > lambda_k
    num_confirmed = 0
    for k in range(len(r_stats)):
        if r_stats[k] > lambdas[k]:
            num_confirmed = k + 1

    return candidate_indices[:num_confirmed]


def run_clinical_etl(
    df: pd.DataFrame,
    raw_bytes: bytes,
    index_col: str,
    ref_col: str,
    diag_col: Optional[str] = None,
    comp_col: Optional[str] = None,
    id_col: Optional[str] = None,
    alpha_outlier: float = 0.05
) -> ETLReport:
    """
    Executes schema parsing, censored flag imputation, unit scale verification,
    and Rosner ESD outlier auditing on uploaded bench data.
    """
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
    work_df = df.copy()
    raw_rows = len(work_df)

    idx_parsed, idx_cens = _parse_clinical_numeric(work_df[index_col])
    ref_parsed, ref_cens = _parse_clinical_numeric(work_df[ref_col])

    work_df["_index_num"] = idx_parsed
    work_df["_ref_num"] = ref_parsed

    censored_records = (
        [{"Column": index_col, **rec} for rec in idx_cens]
        + [{"Column": ref_col, **rec} for rec in ref_cens]
    )

    if comp_col and comp_col in work_df.columns:
        comp_parsed, _ = _parse_clinical_numeric(work_df[comp_col])
        work_df["_comp_num"] = comp_parsed

    if diag_col and diag_col in work_df.columns:
        # Normalize binary labels (0/1, Positive/Negative, Yes/No, True/False)
        diag_series = work_df[diag_col]
        if diag_series.dtype == object or str(diag_series.dtype) == "category":
            mapping = {
                "1": 1, "pos": 1, "positive": 1, "yes": 1, "true": 1, "disease": 1, "case": 1,
                "0": 0, "neg": 0, "negative": 0, "no": 0, "false": 0, "control": 0, "healthy": 0
            }
            work_df["_diag_bin"] = (
                diag_series.astype(str).str.strip().str.lower().map(mapping)
            )
        else:
            work_df["_diag_bin"] = pd.to_numeric(diag_series, errors="coerce")

    required_cols = ["_index_num", "_ref_num"]
    if diag_col and "_diag_bin" in work_df.columns:
        required_cols.append("_diag_bin")

    valid_mask = work_df[required_cols].notna().all(axis=1)
    clean_df = work_df.loc[valid_mask].copy().reset_index(drop=True)
    missing_dropped = raw_rows - len(clean_df)

    # Automated Unit Mismatch Check (e.g., ng/L vs ug/L or mg/dL vs mmol/L)
    nonzero_ref = clean_df["_ref_num"].replace(0, np.nan).dropna()
    nonzero_idx = clean_df.loc[nonzero_ref.index, "_index_num"]
    median_ratio = float(np.median(nonzero_idx / nonzero_ref)) if len(nonzero_ref) > 0 else 1.0

    unit_warning = None
    if median_ratio > 50.0 or (0 < median_ratio < 0.02):
        unit_warning = (
            f"Potential Unit Scale Mismatch Detected: Median Index/Reference ratio is "
            f"{median_ratio:.4g}x (e.g., ng/L vs µg/L or mg/dL vs mmol/L). Verify analyzer units."
        )

    # Outlier Audit on paired differences
    diffs = (clean_df["_index_num"] - clean_df["_ref_num"]).to_numpy()
    outlier_pos = rosner_esd_test(diffs, alpha=alpha_outlier)
    clean_df["_is_outlier"] = False
    if outlier_pos:
        clean_df.loc[outlier_pos, "_is_outlier"] = True

    outlier_cols = [c for c in [id_col, ref_col, index_col] if c and c in clean_df.columns]
    outlier_table = clean_df.loc[outlier_pos, outlier_cols].copy()
    if not outlier_table.empty:
        outlier_table["Difference (Index - Ref)"] = np.round(diffs[outlier_pos], 3)
        outlier_table["Pct_Diff (%)"] = np.round(
            100.0 * diffs[outlier_pos] / np.maximum(1e-9, clean_df.loc[outlier_pos, "_ref_num"]), 2
        )

    return ETLReport(
        raw_rows=raw_rows,
        valid_paired_rows=len(clean_df),
        missing_dropped=missing_dropped,
        censored_parsed=len(censored_records),
        censored_records=censored_records,
        unit_warning=unit_warning,
        median_ratio=median_ratio,
        outlier_indices=outlier_pos,
        outlier_table=outlier_table,
        clean_df=clean_df,
        dataset_sha256=sha256_hash,
    )


# ==============================================================================
# 3. DIAGNOSTIC ACCURACY & EXACT DELONG ROC ENGINE
# ==============================================================================

@dataclass
class DeLongResult:
    auc: float
    se: float
    ci_lower: float
    ci_upper: float
    n_pos: int
    n_neg: int


@dataclass
class PairedDeLongResult:
    auc_1: float
    ci_1: Tuple[float, float]
    auc_2: float
    ci_2: Tuple[float, float]
    auc_diff: float
    se_diff: float
    z_stat: float
    p_value: float


def _delong_structural_components(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes exact DeLong structural components V10 and V01 via midranks
    (Sun & Xu, 2014) in O(N log N) time with exact tie handling.
    """
    m = len(pos_scores)
    n = len(neg_scores)
    combined = np.concatenate([pos_scores, neg_scores])

    tx = stats.rankdata(pos_scores, method="average")
    ty = stats.rankdata(neg_scores, method="average")
    txy = stats.rankdata(combined, method="average")

    v10 = (txy[:m] - tx) / n
    v01 = 1.0 - (txy[m:] - ty) / m
    auc = float(np.mean(v10))
    return auc, v10, v01


def delong_roc_ci(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 0.05) -> DeLongResult:
    """Computes non-parametric AUC and exact DeLong 95% confidence interval."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    m, n = len(pos_scores), len(neg_scores)

    if m < 2 or n < 2:
        raise ValueError("DeLong AUC requires at least 2 positive and 2 negative cases.")

    auc, v10, v01 = _delong_structural_components(pos_scores, neg_scores)
    s10 = np.var(v10, ddof=1)
    s01 = np.var(v01, ddof=1)
    var_auc = (s10 / m) + (s01 / n)
    se = float(np.sqrt(max(var_auc, 1e-12)))

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_low = float(np.clip(auc - z_crit * se, 0.0, 1.0))
    ci_high = float(np.clip(auc + z_crit * se, 0.0, 1.0))

    return DeLongResult(auc=auc, se=se, ci_lower=ci_low, ci_upper=ci_high, n_pos=m, n_neg=n)


def paired_delong_test(
    y_true: np.ndarray,
    score_1: np.ndarray,
    score_2: np.ndarray,
    alpha: float = 0.05
) -> PairedDeLongResult:
    """
    Two-sided correlated DeLong test comparing two diagnostic assays measured on
    the identical patient cohort.
    """
    y_true = np.asarray(y_true, dtype=int)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    m = int(np.sum(pos_mask))
    n = int(np.sum(neg_mask))

    auc1, v10_1, v01_1 = _delong_structural_components(score_1[pos_mask], score_1[neg_mask])
    auc2, v10_2, v01_2 = _delong_structural_components(score_2[pos_mask], score_2[neg_mask])

    s10 = np.cov(np.vstack([v10_1, v10_2]), ddof=1)
    s01 = np.cov(np.vstack([v01_1, v01_2]), ddof=1)
    s = (s10 / m) + (s01 / n)

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    se1 = np.sqrt(max(s[0, 0], 1e-12))
    se2 = np.sqrt(max(s[1, 1], 1e-12))

    var_diff = max(s[0, 0] + s[1, 1] - 2.0 * s[0, 1], 1e-12)
    se_diff = float(np.sqrt(var_diff))
    diff = float(auc1 - auc2)
    z_stat = diff / se_diff
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

    return PairedDeLongResult(
        auc_1=auc1,
        ci_1=(float(np.clip(auc1 - z_crit * se1, 0.0, 1.0)), float(np.clip(auc1 + z_crit * se1, 0.0, 1.0))),
        auc_2=auc2,
        ci_2=(float(np.clip(auc2 - z_crit * se2, 0.0, 1.0)), float(np.clip(auc2 + z_crit * se2, 0.0, 1.0))),
        auc_diff=diff,
        se_diff=se_diff,
        z_stat=float(z_stat),
        p_value=p_val,
    )


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score binomial 95% confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + (z ** 2) / (4.0 * n)) / n)) / denom
    return (float(np.clip(center - half, 0.0, 1.0)), float(np.clip(center + half, 0.0, 1.0)))


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Clopper-Pearson exact binomial 95% confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (low, high)


def evaluate_cutoff(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    prevalence: Optional[float] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Evaluates full 2x2 diagnostic performance at a specific decision cutoff,
    including Wilson & Exact CIs, Simel LR CIs, DOR, and Bayes-adjusted PPV/NPV.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    n_pos = tp + fn
    n_neg = tn + fp
    total = n_pos + n_neg

    sens = tp / n_pos if n_pos > 0 else 0.0
    spec = tn / n_neg if n_neg > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0

    # Haldane-Anscombe 0.5 continuity correction when a cell is zero for log-scale LRs
    tp_c, fn_c, fp_c, tn_c = (
        (tp + 0.5, fn + 0.5, fp + 0.5, tn + 0.5)
        if min(tp, fn, fp, tn) == 0
        else (float(tp), float(fn), float(fp), float(tn))
    )
    sens_c = tp_c / (tp_c + fn_c)
    spec_c = tn_c / (tn_c + fp_c)

    lr_pos = sens_c / (1.0 - spec_c)
    lr_neg = (1.0 - sens_c) / spec_c
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    se_log_lr_pos = np.sqrt((1.0 / tp_c) - (1.0 / (tp_c + fn_c)) + (1.0 / fp_c) - (1.0 / (fp_c + tn_c)))
    lr_pos_ci = (float(np.exp(np.log(lr_pos) - z * se_log_lr_pos)), float(np.exp(np.log(lr_pos) + z * se_log_lr_pos)))

    se_log_lr_neg = np.sqrt((1.0 / fn_c) - (1.0 / (tp_c + fn_c)) + (1.0 / tn_c) - (1.0 / (fp_c + tn_c)))
    lr_neg_ci = (float(np.exp(np.log(lr_neg) - z * se_log_lr_neg)), float(np.exp(np.log(lr_neg) + z * se_log_lr_neg)))

    dor = (tp_c * tn_c) / (fp_c * fn_c)
    se_log_dor = np.sqrt(1.0 / tp_c + 1.0 / fn_c + 1.0 / fp_c + 1.0 / tn_c)
    dor_ci = (float(np.exp(np.log(dor) - z * se_log_dor)), float(np.exp(np.log(dor) + z * se_log_dor)))

    sample_prev = n_pos / total if total > 0 else 0.0
    prev = sample_prev if prevalence is None else float(np.clip(prevalence, 1e-5, 1.0 - 1e-5))

    # Bayes' Theorem for Predictive Values at target prevalence
    ppv_bayes = (sens * prev) / max(1e-12, (sens * prev + (1.0 - spec) * (1.0 - prev)))
    npv_bayes = (spec * (1.0 - prev)) / max(1e-12, (spec * (1.0 - prev) + (1.0 - sens) * prev))

    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens,
        "sens_wilson": wilson_ci(tp, n_pos, alpha),
        "sens_exact": clopper_pearson_ci(tp, n_pos, alpha),
        "specificity": spec,
        "spec_wilson": wilson_ci(tn, n_neg, alpha),
        "spec_exact": clopper_pearson_ci(tn, n_neg, alpha),
        "accuracy": acc,
        "youden_j": sens + spec - 1.0,
        "lr_pos": float(lr_pos),
        "lr_pos_ci": lr_pos_ci,
        "lr_neg": float(lr_neg),
        "lr_neg_ci": lr_neg_ci,
        "dor": float(dor),
        "dor_ci": dor_ci,
        "sample_prevalence": sample_prev,
        "target_prevalence": prev,
        "ppv": float(ppv_bayes),
        "npv": float(npv_bayes),
    }


def compute_roc_and_cutoffs(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Constructs empirical ROC curve and identifies four clinically distinct operating cutoffs:
    1. Youden's J Optimal (Balanced)
    2. Closest-to-(0,1) Euclidean Optimal
    3. Rule-Out / Screening Cutoff (Sensitivity >= 95%, max Specificity)
    4. Rule-In / Confirmatory Cutoff (Specificity >= 95%, max Sensitivity)
    """
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_score)
    # Filter infinite initial threshold produced by scikit-learn
    finite_mask = np.isfinite(thresholds)
    fpr_f = fpr[finite_mask]
    tpr_f = tpr[finite_mask]
    thresh_f = thresholds[finite_mask]
    spec_f = 1.0 - fpr_f

    roc_df = pd.DataFrame({
        "Threshold": thresh_f,
        "Sensitivity": tpr_f,
        "Specificity": spec_f,
        "FPR": fpr_f,
        "Youden_J": tpr_f + spec_f - 1.0,
        "Dist_01": np.sqrt((1.0 - tpr_f) ** 2 + fpr_f ** 2),
    })

    # 1. Youden's J
    idx_youden = int(np.argmax(roc_df["Youden_J"]))
    # 2. Closest to (0,1)
    idx_dist = int(np.argmin(roc_df["Dist_01"]))

    # 3. Rule-Out (Sensitivity >= 0.95, maximize Specificity)
    rule_out_candidates = roc_df[roc_df["Sensitivity"] >= 0.95]
    if not rule_out_candidates.empty:
        idx_rule_out = int(rule_out_candidates["Specificity"].idxmax())
    else:
        idx_rule_out = int(roc_df["Sensitivity"].idxmax())

    # 4. Rule-In (Specificity >= 0.95, maximize Sensitivity)
    rule_in_candidates = roc_df[roc_df["Specificity"] >= 0.95]
    if not rule_in_candidates.empty:
        idx_rule_in = int(rule_in_candidates["Sensitivity"].idxmax())
    else:
        idx_rule_in = int(roc_df["Specificity"].idxmax())

    cutoffs = {
        "Youden's J (Balanced)": float(roc_df.loc[idx_youden, "Threshold"]),
        "Closest-to-(0,1)": float(roc_df.loc[idx_dist, "Threshold"]),
        "Rule-Out (Sens ≥ 95%)": float(roc_df.loc[idx_rule_out, "Threshold"]),
        "Rule-In (Spec ≥ 95%)": float(roc_df.loc[idx_rule_in, "Threshold"]),
    }
    return roc_df, cutoffs


# ==============================================================================
# 4. METHOD COMPARISON & AGREEMENT ENGINE (CLSI EP09)
# ==============================================================================

@dataclass
class PassingBablokResult:
    slope: float
    slope_ci: Tuple[float, float]
    intercept: float
    intercept_ci: Tuple[float, float]
    cusum_h: float
    cusum_critical: float
    is_linear: bool
    n_pairs_used: int


@dataclass
class DemingResult:
    slope: float
    slope_ci: Tuple[float, float]
    intercept: float
    intercept_ci: Tuple[float, float]
    error_ratio: float
    pearson_r: float
    lin_ccc: float


@dataclass
class BlandAltmanResult:
    mode: str
    mean_bias: float
    mean_bias_ci: Tuple[float, float]
    sd_diff: float
    loa_lower: float
    loa_lower_ci: Tuple[float, float]
    loa_upper: float
    loa_upper_ci: Tuple[float, float]
    prop_slope: float
    prop_p_value: float
    means: np.ndarray
    diffs: np.ndarray


def passing_bablok_regression(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> PassingBablokResult:
    """
    Exact non-parametric Passing-Bablok regression (1983) with K-offset for negative
    slopes, exact confidence bounds, and CUSUM linearity test.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples = len(x)

    if n_samples < 10:
        raise ValueError("Passing-Bablok regression requires at least 10 paired samples.")

    # Cap O(N^2) pairwise expansion for ultra-large cohorts via deterministic subsampling
    if n_samples > 2500:
        rng = np.random.default_rng(42)
        sub_idx = np.sort(rng.choice(n_samples, size=2500, replace=False))
        x = x[sub_idx]
        y = y[sub_idx]
        n_samples = 2500

    i_idx, j_idx = np.triu_indices(n_samples, k=1)
    dx = x[j_idx] - x[i_idx]
    dy = y[j_idx] - y[i_idx]

    # Exclude identical pairs (dx == 0 and dy == 0)
    non_identical = ~((dx == 0) & (dy == 0))
    dx = dx[non_identical]
    dy = dy[non_identical]

    slopes = np.empty(len(dx), dtype=float)
    zero_dx = dx == 0
    slopes[~zero_dx] = dy[~zero_dx] / dx[~zero_dx]
    slopes[zero_dx & (dy > 0)] = np.inf
    slopes[zero_dx & (dy < 0)] = -np.inf

    # Exclude exact -1 slopes per Passing-Bablok specification
    slopes = slopes[slopes != -1.0]
    slopes.sort()

    n_valid = len(slopes)
    k_offset = int(np.sum(slopes < -1.0))

    # Shifted median slope
    total_idx = n_valid + k_offset
    if total_idx % 2 == 1:
        mid = (total_idx - 1) // 2
        slope = float(slopes[min(mid, n_valid - 1)])
    else:
        m_low = max(0, (total_idx // 2) - 1)
        m_high = min(n_valid - 1, total_idx // 2)
        slope = float(0.5 * (slopes[m_low] + slopes[m_high]))

    # Exact 95% CI indices for slope
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    c_alpha = z_crit * np.sqrt((n_samples * (n_samples - 1) * (2 * n_samples + 5)) / 18.0)
    m1 = int(np.round((n_valid - c_alpha) / 2.0))
    m2 = n_valid - m1 + 1

    idx_low = int(np.clip(m1 - 1 + k_offset, 0, n_valid - 1))
    idx_high = int(np.clip(m2 - 1 + k_offset, 0, n_valid - 1))
    slope_low = float(slopes[idx_low])
    slope_high = float(slopes[idx_high])

    # Intercept & 95% CI
    intercept = float(np.median(y - slope * x))
    intercept_low = float(np.median(y - slope_high * x))
    intercept_high = float(np.median(y - slope_low * x))

    # CUSUM Test for Linearity (Passing & Bablok 1983)
    residuals = y - (intercept + slope * x)
    n_pos_res = int(np.sum(residuals > 0))
    n_neg_res = int(np.sum(residuals < 0))
    scores_r = np.zeros_like(residuals)
    if n_pos_res > 0 and n_neg_res > 0:
        scores_r[residuals > 0] = np.sqrt(n_neg_res / n_pos_res)
        scores_r[residuals < 0] = -np.sqrt(n_pos_res / n_neg_res)

    d_rank = (x + y / slope) if abs(slope) > 1e-9 else x
    order = np.argsort(d_rank)
    cusum = np.cumsum(scores_r[order])
    cusum_h = float(np.max(np.abs(cusum)) / np.sqrt(n_pos_res + n_neg_res)) if (n_pos_res + n_neg_res) > 0 else 0.0
    cusum_crit = 1.36  # Exact Kolmogorov-Smirnov critical constant at alpha = 0.05

    return PassingBablokResult(
        slope=slope,
        slope_ci=(slope_low, slope_high),
        intercept=intercept,
        intercept_ci=(intercept_low, intercept_high),
        cusum_h=cusum_h,
        cusum_critical=cusum_crit,
        is_linear=(cusum_h <= cusum_crit),
        n_pairs_used=n_valid,
    )


def _deming_fit(x: np.ndarray, y: np.ndarray, error_ratio: float) -> Tuple[float, float]:
    sxx = np.var(x, ddof=1)
    syy = np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    if abs(sxy) < 1e-12:
        return 1.0, float(np.mean(y) - np.mean(x))
    term = syy - error_ratio * sxx
    b = (term + np.sqrt(term ** 2 + 4.0 * error_ratio * (sxy ** 2))) / (2.0 * sxy)
    a = np.mean(y) - b * np.mean(x)
    return float(b), float(a)


def deming_regression(
    x: np.ndarray,
    y: np.ndarray,
    error_ratio: float = 1.0,
    alpha: float = 0.05
) -> DemingResult:
    """
    Deming (errors-in-variables) regression with Jackknife 95% confidence intervals
    and Lin's Concordance Correlation Coefficient (CCC).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 10:
        raise ValueError("Deming regression requires at least 10 paired samples.")

    slope, intercept = _deming_fit(x, y, error_ratio)

    # Leave-one-out Jackknife standard errors (CLSI EP09 / mcr standard)
    jk_slopes = np.empty(n, dtype=float)
    jk_intercepts = np.empty(n, dtype=float)
    for i in range(n):
        xi = np.delete(x, i)
        yi = np.delete(y, i)
        jk_slopes[i], jk_intercepts[i] = _deming_fit(xi, yi, error_ratio)

    se_slope = np.sqrt(((n - 1) / n) * np.sum((jk_slopes - np.mean(jk_slopes)) ** 2))
    se_intercept = np.sqrt(((n - 1) / n) * np.sum((jk_intercepts - np.mean(jk_intercepts)) ** 2))

    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 2)
    slope_ci = (float(slope - t_crit * se_slope), float(slope + t_crit * se_slope))
    intercept_ci = (float(intercept - t_crit * se_intercept), float(intercept + t_crit * se_intercept))

    r_val = float(stats.pearsonr(x, y)[0])
    # Lin's Concordance Correlation Coefficient
    sxx = np.var(x, ddof=1)
    syy = np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    lin_ccc = float((2.0 * sxy) / (sxx + syy + (np.mean(x) - np.mean(y)) ** 2))

    return DemingResult(
        slope=slope,
        slope_ci=slope_ci,
        intercept=intercept,
        intercept_ci=intercept_ci,
        error_ratio=error_ratio,
        pearson_r=r_val,
        lin_ccc=lin_ccc,
    )


def bland_altman_analysis(
    x: np.ndarray,
    y: np.ndarray,
    mode: str = "absolute",
    alpha: float = 0.05
) -> BlandAltmanResult:
    """
    Bland-Altman agreement analysis with exact 95% confidence bounds around
    Mean Bias and Upper/Lower Limits of Agreement (Bland & Altman, 1999).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    means = 0.5 * (x + y)

    if mode == "percentage":
        safe_means = np.where(np.abs(means) < 1e-9, np.nan, means)
        diffs = 100.0 * (y - x) / safe_means
        valid = ~np.isnan(diffs)
        means = means[valid]
        diffs = diffs[valid]
        n = len(diffs)
    else:
        diffs = y - x

    mean_bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    loa_lower = mean_bias - 1.96 * sd_diff
    loa_upper = mean_bias + 1.96 * sd_diff

    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=max(1, n - 1))
    se_bias = sd_diff / np.sqrt(n)
    se_loa = np.sqrt((3.0 * (sd_diff ** 2)) / n)

    # Proportional bias regression (Differences vs Means)
    reg = stats.linregress(means, diffs)

    return BlandAltmanResult(
        mode=mode,
        mean_bias=mean_bias,
        mean_bias_ci=(float(mean_bias - t_crit * se_bias), float(mean_bias + t_crit * se_bias)),
        sd_diff=sd_diff,
        loa_lower=float(loa_lower),
        loa_lower_ci=(float(loa_lower - t_crit * se_loa), float(loa_lower + t_crit * se_loa)),
        loa_upper=float(loa_upper),
        loa_upper_ci=(float(loa_upper - t_crit * se_loa), float(loa_upper + t_crit * se_loa)),
        prop_slope=float(reg.slope),
        prop_p_value=float(reg.pvalue),
        means=means,
        diffs=diffs,
    )


def estimate_mdp_bias(
    mdp_list: List[float],
    slope: float,
    slope_ci: Tuple[float, float],
    intercept: float,
    intercept_ci: Tuple[float, float]
) -> pd.DataFrame:
    """
    Computes predicted Index assay values and systematic bias (absolute & relative %)
    at clinical Medical Decision Points (MDPs).
    """
    rows = []
    for mdp in mdp_list:
        pred = intercept + slope * mdp
        pred_low = intercept_ci[0] + slope_ci[0] * mdp
        pred_high = intercept_ci[1] + slope_ci[1] * mdp
        abs_bias = pred - mdp
        rel_bias = (abs_bias / mdp * 100.0) if abs(mdp) > 1e-9 else np.nan
        rows.append({
            "Medical Decision Point (Ref)": round(mdp, 3),
            "Predicted Index Value": round(pred, 3),
            "95% CI Lower": round(min(pred_low, pred_high), 3),
            "95% CI Upper": round(max(pred_low, pred_high), 3),
            "Systematic Bias (Abs)": round(abs_bias, 3),
            "Systematic Bias (%)": round(rel_bias, 2),
        })
    return pd.DataFrame(rows)


def get_runtime_manifest() -> Dict[str, str]:
    """Returns exact runtime library versions for regulatory audit reproducibility."""
    return {
        "Timestamp_UTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Python": platform.python_version(),
        "NumPy": np.__version__,
        "SciPy": scipy.__version__,
        "Pandas": pd.__version__,
        "Scikit-Learn": sklearn.__version__,
  }
