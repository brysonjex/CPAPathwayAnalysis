import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

matplotlib.use("Agg")

DATA_FILE = "Alternative CPA Pathways Survey_December 31, 2025_09.45.csv"
OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")


@dataclass
class ColumnMetadata:
    headers: List[str]
    question_text: Dict[str, str]


@dataclass
class InterestMapping:
    mapping: Dict[str, float]
    scale_type: str
    high_interest_threshold: float


def load_column_metadata(path: str) -> ColumnMetadata:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        question_row = next(reader)
        _ = next(reader)
    question_text = {header: text for header, text in zip(headers, question_row)}
    return ColumnMetadata(headers=headers, question_text=question_text)


def read_survey_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=[1, 2])


def find_best_column(columns: List[str], question_text: Dict[str, str], patterns: List[str]) -> Optional[str]:
    scores: List[Tuple[int, str]] = []
    for column in columns:
        text = f"{column} {question_text.get(column, '')}".lower()
        score = sum(1 for pat in patterns if re.search(pat, text, flags=re.IGNORECASE))
        scores.append((score, column))
    scores.sort(reverse=True)
    top_score, top_column = scores[0]
    return top_column if top_score > 0 else None


def normalize_academic_level(raw_value: str) -> Optional[str]:
    if pd.isna(raw_value):
        return None
    text = str(raw_value).strip().lower()
    undergraduate_tokens = [
        "freshman",
        "sophomore",
        "junior",
        "senior",
        "bachelor",
        "undergraduate",
    ]
    graduate_tokens = [
        "master",
        "mba",
        "macc",
        "phd",
        "doctoral",
        "graduate",
    ]
    if any(token in text for token in undergraduate_tokens):
        return "Undergraduate"
    if any(token in text for token in graduate_tokens):
        return "Graduate"
    return None


def infer_interest_mapping(series: pd.Series) -> InterestMapping:
    cleaned = series.dropna().astype(str).str.strip()
    lowered = cleaned.str.lower()
    unique_values = sorted(set(lowered.tolist()))

    yes_no = {"yes": 1.0, "no": 0.0}
    if unique_values and all(value in yes_no for value in unique_values):
        return InterestMapping(mapping=yes_no, scale_type="binary", high_interest_threshold=1.0)

    likert_scale = {
        "very unlikely": 1.0,
        "somewhat unlikely": 2.0,
        "neither likely nor unlikely": 3.0,
        "somewhat likely": 4.0,
        "very likely": 5.0,
    }
    if any(value in likert_scale for value in unique_values):
        return InterestMapping(mapping=likert_scale, scale_type="likert", high_interest_threshold=4.0)

    numeric_values = pd.to_numeric(cleaned, errors="coerce")
    if numeric_values.notna().any():
        return InterestMapping(mapping={}, scale_type="numeric", high_interest_threshold=float(numeric_values.quantile(0.75)))

    return InterestMapping(mapping={}, scale_type="unknown", high_interest_threshold=1.0)


def apply_interest_mapping(series: pd.Series, mapping: InterestMapping) -> pd.Series:
    if mapping.scale_type in {"binary", "likert"}:
        return series.astype(str).str.strip().str.lower().map(mapping.mapping)
    return pd.to_numeric(series, errors="coerce")


def summarize_by_group(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["high_interest"] = df["cpa_interest_numeric"] >= threshold
    summary = (
        df.groupby("academic_level_group")["cpa_interest_numeric"]
        .agg(["count", "mean", "median", "std"])
        .rename(columns={"count": "n", "std": "std_dev"})
    )
    high_interest = df.groupby("academic_level_group")["high_interest"].mean() * 100
    summary["pct_high_interest"] = high_interest
    return summary.reset_index()


def compare_groups(df: pd.DataFrame, scale_type: str) -> Dict[str, float]:
    group_a = df[df["academic_level_group"] == "Undergraduate"]["cpa_interest_numeric"].dropna()
    group_b = df[df["academic_level_group"] == "Graduate"]["cpa_interest_numeric"].dropna()

    results: Dict[str, float] = {
        "n_undergraduate": int(group_a.shape[0]),
        "n_graduate": int(group_b.shape[0]),
    }

    if scale_type == "binary":
        contingency = pd.crosstab(df["academic_level_group"], df["cpa_interest_numeric"])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        n_total = contingency.to_numpy().sum()
        phi = np.sqrt(chi2 / n_total) if n_total else np.nan
        results.update(
            {
                "test": "chi_square",
                "test_statistic": float(chi2),
                "p_value": float(p_value),
                "effect_size": float(phi),
            }
        )
        return results

    use_shapiro = all(3 <= len(group) <= 5000 for group in [group_a, group_b])
    if use_shapiro:
        _, p_a = stats.shapiro(group_a)
        _, p_b = stats.shapiro(group_b)
        normal = p_a > 0.05 and p_b > 0.05
    else:
        normal = False

    if normal:
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False, nan_policy="omit")
        pooled_std = np.sqrt((group_a.var(ddof=1) + group_b.var(ddof=1)) / 2)
        d = (group_b.mean() - group_a.mean()) / pooled_std if pooled_std else np.nan
        results.update(
            {
                "test": "t_test",
                "test_statistic": float(t_stat),
                "p_value": float(p_value),
                "effect_size": float(d),
            }
        )
        return results

    u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    z_value = stats.norm.ppf(p_value / 2) * (-1)
    r = z_value / np.sqrt(group_a.size + group_b.size)
    results.update(
        {
            "test": "mann_whitney_u",
            "test_statistic": float(u_stat),
            "p_value": float(p_value),
            "effect_size": float(r),
        }
    )
    return results


def interpret_results(results: Dict[str, float]) -> str:
    p_value = results.get("p_value")
    if p_value is None or np.isnan(p_value):
        return "Statistical comparison could not be performed due to insufficient data."
    if p_value < 0.05:
        return "Graduate students show statistically different interest in CPA compared to undergraduates."
    return "No statistically significant difference in CPA interest between graduate and undergraduate students."


def generate_plots(summary: pd.DataFrame, df: pd.DataFrame, threshold: float) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.barplot(x="academic_level_group", y="mean", data=summary, palette="Set2")
    plt.ylabel("Mean CPA Interest")
    plt.xlabel("Academic Level")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "mean_cpa_interest_by_level.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(x="academic_level_group", y="cpa_interest_numeric", data=df, palette="Set3")
    plt.ylabel("CPA Interest")
    plt.xlabel("Academic Level")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cpa_interest_distribution_by_level.png"), dpi=300)
    plt.close()

    pct_data = summary[["academic_level_group", "pct_high_interest"]]
    plt.figure(figsize=(6, 4))
    sns.barplot(x="academic_level_group", y="pct_high_interest", data=pct_data, palette="Set1")
    plt.ylabel(f"% High Interest (>= {threshold})")
    plt.xlabel("Academic Level")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pct_high_interest_by_level.png"), dpi=300)
    plt.close()


def main() -> None:
    metadata = load_column_metadata(DATA_FILE)
    df = read_survey_data(DATA_FILE)

    print("Column names:", df.columns.tolist())
    print("Data types:\n", df.dtypes)
    print("Missing values:\n", df.isna().sum())

    academic_column = find_best_column(
        df.columns.tolist(), metadata.question_text, [r"undergraduate", r"graduate", r"student"]
    )
    interest_column = find_best_column(
        df.columns.tolist(), metadata.question_text, [r"cpa", r"pursue", r"likely"]
    )

    if not academic_column or not interest_column:
        raise ValueError("Unable to automatically map academic level or CPA interest column.")

    df["academic_level_group"] = df[academic_column].apply(normalize_academic_level)
    interest_mapping = infer_interest_mapping(df[interest_column])
    df["cpa_interest_numeric"] = apply_interest_mapping(df[interest_column], interest_mapping)

    cleaned = df.dropna(subset=["academic_level_group", "cpa_interest_numeric"]).copy()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cleaned.to_csv(os.path.join(OUTPUT_DIR, "cleaned_cpa_survey.csv"), index=False)

    summary = summarize_by_group(cleaned, interest_mapping.high_interest_threshold)
    summary.to_csv(os.path.join(OUTPUT_DIR, "cpa_interest_summary_by_level.csv"), index=False)

    stats_results = compare_groups(cleaned, interest_mapping.scale_type)
    stats_results["interpretation"] = interpret_results(stats_results)
    stats_results["academic_level_column"] = academic_column
    stats_results["cpa_interest_column"] = interest_column
    stats_results["scale_type"] = interest_mapping.scale_type
    stats_results["high_interest_threshold"] = interest_mapping.high_interest_threshold

    pd.DataFrame([stats_results]).to_csv(
        os.path.join(OUTPUT_DIR, "cpa_interest_statistical_comparison.csv"), index=False
    )
    with open(os.path.join(OUTPUT_DIR, "cpa_interest_statistical_comparison.json"), "w") as handle:
        json.dump(stats_results, handle, indent=2)

    generate_plots(summary, cleaned, interest_mapping.high_interest_threshold)


if __name__ == "__main__":
    main()
