from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# PaperPulse AI - LLM Result EDA Batch
# -----------------------------------------------------------------------------
# This script does not run a local LLM.
# It reads CSV files already created by monthly_llm_eda.py and creates
# dashboard-ready aggregate files.
#
# Input directory, by default:
#   data/eda_llm/
#
# Output directory, by default:
#   data/eda_llm_derived/
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LLM_DIR = PROJECT_ROOT / "data" / "eda_llm"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eda_llm_derived"

REQUIRED_OR_USEFUL_FILES = {
    "paper_llm_insights": "paper_llm_insights.csv",
    "paper_paradigm_labels": "paper_paradigm_labels.csv",
    "structured_paper_insights": "structured_paper_insights.csv",
    "paper_concepts": "paper_concepts.csv",
    "concept_nodes": "concept_nodes.csv",
    "concept_edges": "concept_edges.csv",
    "monthly_paradigm_stats": "monthly_paradigm_stats.csv",
    "monthly_concept_stats": "monthly_concept_stats.csv",
    "llm_run_log": "llm_run_log.csv",
}

TEXT_COLUMNS = [
    "canonical_arxiv_id",
    "title",
    "summary",
    "authors_text",
    "primary_category",
    "categories_text",
    "published_month",
    "primary_paradigm",
    "secondary_paradigms",
    "contribution_type",
    "problem",
    "method",
    "result",
    "keywords",
    "concept_name",
    "concept_type",
    "abs_url",
    "pdf_url",
]

PARADIGM_ORDER = [
    "RAG/Retrieval",
    "Agent/Tool Use",
    "Alignment/Safety",
    "Reasoning",
    "Efficiency/Optimization",
    "Multimodal",
    "Benchmark/Evaluation",
    "Theory/Methodology",
    "Application",
    "Dataset/Benchmark",
    "Other",
]


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create derived EDA files from monthly local LLM analysis results."
    )

    parser.add_argument(
        "--llm-dir",
        type=str,
        default=str(DEFAULT_LLM_DIR),
        help="Directory containing monthly_llm_eda.py outputs, default=data/eda_llm",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save derived LLM EDA files, default=data/eda_llm_derived",
    )
    parser.add_argument(
        "--target-month",
        type=str,
        default=None,
        help="Optional YYYY-MM filter. Example: 2026-07",
    )
    parser.add_argument(
        "--recent-months",
        type=int,
        default=0,
        help="If > 0 and --target-month is omitted, keep only latest N months.",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.50,
        help="Threshold for low confidence paper list.",
    )
    parser.add_argument(
        "--top-n-concepts",
        type=int,
        default=100,
        help="Maximum rows retained for concept stats.",
    )
    parser.add_argument(
        "--top-n-keywords",
        type=int,
        default=100,
        help="Maximum rows retained for LLM keyword stats.",
    )
    parser.add_argument(
        "--top-n-edges",
        type=int,
        default=300,
        help="Maximum rows retained for concept edge stats.",
    )

    return parser.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return pd.DataFrame()


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "", "NaT": ""}).str.strip()

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    for col in ["paper_count", "edge_weight", "rank", "rank_in_month", "month_total", "ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_llm_outputs(llm_dir: Path) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    for key, filename in REQUIRED_OR_USEFUL_FILES.items():
        path = llm_dir / filename
        df = normalize_df(read_csv_if_exists(path))
        outputs[key] = df
        print(f"[INFO] Loaded {filename}: {len(df)} rows | exists={path.exists()}")

    return outputs


def merge_fill_on_id(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty and right.empty:
        return pd.DataFrame()
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()
    if "canonical_arxiv_id" not in left.columns or "canonical_arxiv_id" not in right.columns:
        return left.copy()

    left = left.copy()
    right = right.copy().drop_duplicates(subset=["canonical_arxiv_id"], keep="last")

    merged = left.merge(right, on="canonical_arxiv_id", how="outer", suffixes=("", "__right"))

    right_cols = [col for col in right.columns if col != "canonical_arxiv_id"]

    for col in right_cols:
        right_col = f"{col}__right"
        if right_col not in merged.columns:
            continue

        if col not in merged.columns:
            merged[col] = merged[right_col]
        else:
            left_nonempty = merged[col].notna() & (merged[col].astype(str).str.strip() != "")
            merged[col] = merged[col].where(left_nonempty, merged[right_col])

        merged = merged.drop(columns=[right_col])

    return merged


def build_master_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base_candidates = [
        data.get("paper_llm_insights", pd.DataFrame()),
        data.get("structured_paper_insights", pd.DataFrame()),
        data.get("paper_paradigm_labels", pd.DataFrame()),
    ]

    master = pd.DataFrame()

    for candidate in base_candidates:
        if candidate is not None and not candidate.empty and "canonical_arxiv_id" in candidate.columns:
            if master.empty:
                master = candidate.copy()
            else:
                master = merge_fill_on_id(master, candidate)

    if master.empty:
        return pd.DataFrame()

    master = normalize_df(master)
    master = master.drop_duplicates(subset=["canonical_arxiv_id"], keep="last")

    for col in ["primary_paradigm", "contribution_type", "primary_category", "published_month"]:
        if col not in master.columns:
            master[col] = ""

    if "confidence" not in master.columns:
        master["confidence"] = np.nan

    # Normalize categories and labels.
    master["primary_paradigm"] = master["primary_paradigm"].replace({"": "Other"})
    master["contribution_type"] = master["contribution_type"].replace({"": "other"})
    master["primary_category"] = master["primary_category"].replace({"": "unknown"})

    if "published_at" in master.columns:
        parsed_dt = pd.to_datetime(master["published_at"], errors="coerce", utc=True)
        if "published_month" not in master.columns or master["published_month"].eq("").all():
            master["published_month"] = parsed_dt.dt.to_period("M").astype(str)
        else:
            missing_month = master["published_month"].eq("") | master["published_month"].isna()
            master.loc[missing_month, "published_month"] = parsed_dt[missing_month].dt.to_period("M").astype(str)

    master["published_month"] = master["published_month"].replace({"NaT": "", "nan": ""})

    # Ensure text columns for the paper explorer.
    for col in ["title", "summary", "problem", "method", "result", "abs_url", "pdf_url"]:
        if col not in master.columns:
            master[col] = ""

    return master


def split_list_field(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    # JSON array string support.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass

    if " | " in text:
        parts = text.split(" | ")
    elif "|" in text:
        parts = text.split("|")
    elif "," in text:
        parts = text.split(",")
    else:
        parts = [text]

    return [part.strip() for part in parts if part.strip()]


def normalize_keyword(keyword: str) -> str:
    keyword = str(keyword).lower().strip()
    keyword = re.sub(r"\s+", " ", keyword)
    keyword = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", keyword)
    return keyword


def parse_concepts_json(value: Any) -> list[dict[str, str]]:
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw_items = json.loads(text)
        except Exception:
            return []

    concepts = []
    if not isinstance(raw_items, list):
        return concepts

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        name = normalize_keyword(item.get("name", ""))
        concept_type = str(item.get("type", "other")).strip().lower() or "other"
        if name:
            concepts.append({"name": name, "type": concept_type})
    return concepts


def build_paper_concepts(data: dict[str, pd.DataFrame], master: pd.DataFrame) -> pd.DataFrame:
    explicit = data.get("paper_concepts", pd.DataFrame())

    if explicit is not None and not explicit.empty:
        concept_df = explicit.copy()
        if "concept_name" not in concept_df.columns and "name" in concept_df.columns:
            concept_df = concept_df.rename(columns={"name": "concept_name"})
        if "concept_type" not in concept_df.columns and "type" in concept_df.columns:
            concept_df = concept_df.rename(columns={"type": "concept_type"})
        for col in ["concept_name", "concept_type"]:
            if col not in concept_df.columns:
                concept_df[col] = ""
        concept_df["concept_name"] = concept_df["concept_name"].map(normalize_keyword)
        concept_df["concept_type"] = concept_df["concept_type"].replace({"": "other"}).str.lower()
    else:
        rows = []
        if "concepts_json" in master.columns:
            for _, row in master.iterrows():
                arxiv_id = row.get("canonical_arxiv_id", "")
                for concept in parse_concepts_json(row.get("concepts_json", "")):
                    rows.append(
                        {
                            "canonical_arxiv_id": arxiv_id,
                            "concept_name": concept["name"],
                            "concept_type": concept["type"],
                        }
                    )
        concept_df = pd.DataFrame(rows)

    if concept_df.empty:
        return pd.DataFrame(
            columns=[
                "canonical_arxiv_id",
                "published_month",
                "primary_category",
                "primary_paradigm",
                "concept_name",
                "concept_type",
            ]
        )

    meta_cols = ["canonical_arxiv_id", "published_month", "primary_category", "primary_paradigm"]
    meta_cols = [col for col in meta_cols if col in master.columns]
    meta = master[meta_cols].drop_duplicates(subset=["canonical_arxiv_id"], keep="last")

    concept_df = concept_df.merge(meta, on="canonical_arxiv_id", how="left", suffixes=("", "__meta"))

    for col in ["published_month", "primary_category", "primary_paradigm"]:
        meta_col = f"{col}__meta"
        if meta_col in concept_df.columns:
            if col not in concept_df.columns:
                concept_df[col] = concept_df[meta_col]
            else:
                concept_df[col] = concept_df[col].where(
                    concept_df[col].notna() & (concept_df[col].astype(str).str.strip() != ""),
                    concept_df[meta_col],
                )
            concept_df = concept_df.drop(columns=[meta_col])

    concept_df = normalize_df(concept_df)
    concept_df = concept_df[concept_df["concept_name"] != ""].copy()
    concept_df = concept_df.drop_duplicates(subset=["canonical_arxiv_id", "concept_name"], keep="last")

    return concept_df


def filter_by_month(
    master: pd.DataFrame,
    concept_df: pd.DataFrame,
    target_month: str | None,
    recent_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if master.empty:
        return master, concept_df, []

    months = sorted(
        month
        for month in master["published_month"].dropna().astype(str).unique().tolist()
        if re.fullmatch(r"\d{4}-\d{2}", month)
    )

    if target_month:
        if not re.fullmatch(r"\d{4}-\d{2}", target_month):
            raise ValueError("--target-month must be YYYY-MM")
        selected_months = [target_month]
    elif recent_months and recent_months > 0:
        selected_months = months[-recent_months:]
    else:
        selected_months = months

    if selected_months:
        master = master[master["published_month"].isin(selected_months)].copy()
        concept_df = concept_df[concept_df["published_month"].isin(selected_months)].copy()

    return master, concept_df, selected_months


def add_ratio(df: pd.DataFrame, count_col: str = "paper_count", ratio_col: str = "ratio") -> pd.DataFrame:
    if df.empty or count_col not in df.columns:
        return df.copy()
    df = df.copy()
    total = df[count_col].sum()
    if total == 0:
        df[ratio_col] = 0.0
    else:
        df[ratio_col] = (df[count_col] / total * 100).round(2)
    return df


def top_joined_values(values: pd.Series, limit: int = 5) -> str:
    counts = Counter(v for v in values.dropna().astype(str).tolist() if v and v != "nan")
    return " | ".join(item for item, _ in counts.most_common(limit))


def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    value = pd.to_numeric(series, errors="coerce").mean()
    if pd.isna(value):
        return 0.0
    return float(round(value, 4))


# -----------------------------------------------------------------------------
# EDA builders
# -----------------------------------------------------------------------------


def create_kpi_summary(
    master: pd.DataFrame,
    concept_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    selected_months: list[str],
    low_confidence_threshold: float,
) -> pd.DataFrame:
    total_papers = int(master["canonical_arxiv_id"].nunique()) if not master.empty else 0
    latest_month = max(selected_months) if selected_months else ""

    top_paradigm = "-"
    top_paradigm_count = 0
    if not master.empty and "primary_paradigm" in master.columns:
        counts = master["primary_paradigm"].value_counts()
        if not counts.empty:
            top_paradigm = str(counts.index[0])
            top_paradigm_count = int(counts.iloc[0])

    top_concept = "-"
    top_concept_count = 0
    if not concept_df.empty:
        counts = concept_df.groupby("concept_name")["canonical_arxiv_id"].nunique().sort_values(ascending=False)
        if not counts.empty:
            top_concept = str(counts.index[0])
            top_concept_count = int(counts.iloc[0])

    avg_confidence = safe_mean(master["confidence"]) if "confidence" in master.columns else 0.0
    low_confidence_count = 0
    if "confidence" in master.columns:
        low_confidence_count = int((pd.to_numeric(master["confidence"], errors="coerce") < low_confidence_threshold).sum())

    rows = [
        {
            "metric_name": "llm_total_papers",
            "metric_value": total_papers,
            "metric_description": "LLM 분석이 완료된 논문 수",
        },
        {
            "metric_name": "llm_months_covered",
            "metric_value": len(selected_months),
            "metric_description": "LLM EDA 대상 월 개수",
        },
        {
            "metric_name": "llm_latest_month",
            "metric_value": latest_month,
            "metric_description": "LLM EDA 대상 중 가장 최근 월",
        },
        {
            "metric_name": "llm_top_paradigm",
            "metric_value": top_paradigm,
            "metric_description": "가장 많이 분류된 연구 패러다임",
        },
        {
            "metric_name": "llm_top_paradigm_count",
            "metric_value": top_paradigm_count,
            "metric_description": "최상위 연구 패러다임 논문 수",
        },
        {
            "metric_name": "llm_avg_confidence",
            "metric_value": avg_confidence,
            "metric_description": "LLM 분류 confidence 평균",
        },
        {
            "metric_name": "llm_low_confidence_count",
            "metric_value": low_confidence_count,
            "metric_description": f"confidence < {low_confidence_threshold} 논문 수",
        },
        {
            "metric_name": "llm_total_concepts",
            "metric_value": int(concept_df["concept_name"].nunique()) if not concept_df.empty else 0,
            "metric_description": "LLM이 추출한 고유 핵심 개념 수",
        },
        {
            "metric_name": "llm_top_concept",
            "metric_value": top_concept,
            "metric_description": "가장 많은 논문에서 추출된 핵심 개념",
        },
        {
            "metric_name": "llm_top_concept_count",
            "metric_value": top_concept_count,
            "metric_description": "최상위 핵심 개념이 등장한 논문 수",
        },
        {
            "metric_name": "llm_concept_edges",
            "metric_value": len(edge_df) if edge_df is not None else 0,
            "metric_description": "Concept co-occurrence edge 수",
        },
    ]

    return pd.DataFrame(rows)


def create_paradigm_stats(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["primary_paradigm", "paper_count", "avg_confidence", "ratio", "rank"])

    stats = (
        master.groupby("primary_paradigm")
        .agg(
            paper_count=("canonical_arxiv_id", "nunique"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    stats["avg_confidence"] = stats["avg_confidence"].round(4)
    stats = add_ratio(stats)
    stats = stats.sort_values("paper_count", ascending=False).reset_index(drop=True)
    stats["rank"] = stats.index + 1
    return stats


def create_monthly_paradigm_stats(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "primary_paradigm",
                "paper_count",
                "month_total",
                "ratio",
                "avg_confidence",
                "previous_month_count",
                "growth_rate",
            ]
        )

    stats = (
        master.dropna(subset=["published_month"])
        .groupby(["published_month", "primary_paradigm"])
        .agg(
            paper_count=("canonical_arxiv_id", "nunique"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        .rename(columns={"published_month": "month"})
    )
    stats["avg_confidence"] = stats["avg_confidence"].round(4)

    totals = stats.groupby("month")["paper_count"].sum().reset_index().rename(columns={"paper_count": "month_total"})
    stats = stats.merge(totals, on="month", how="left")
    stats["ratio"] = (stats["paper_count"] / stats["month_total"] * 100).round(2)
    stats = stats.sort_values(["primary_paradigm", "month"])
    stats["previous_month_count"] = stats.groupby("primary_paradigm")["paper_count"].shift(1).fillna(0).astype(int)
    stats["growth_rate"] = np.where(
        stats["previous_month_count"] > 0,
        (stats["paper_count"] - stats["previous_month_count"]) / stats["previous_month_count"] * 100,
        np.where(stats["paper_count"] > 0, 100.0, 0.0),
    )
    stats["growth_rate"] = pd.Series(stats["growth_rate"]).replace([np.inf, -np.inf], 0).fillna(0).round(2)
    stats = stats.sort_values(["month", "paper_count"], ascending=[True, False])
    return stats


def create_contribution_stats(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["contribution_type", "paper_count", "ratio", "rank"])

    stats = (
        master.groupby("contribution_type")
        .agg(paper_count=("canonical_arxiv_id", "nunique"))
        .reset_index()
    )
    stats = add_ratio(stats)
    stats = stats.sort_values("paper_count", ascending=False).reset_index(drop=True)
    stats["rank"] = stats.index + 1
    return stats


def create_category_paradigm_matrix(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["primary_category", "primary_paradigm", "paper_count", "category_total", "ratio_by_category"])

    stats = (
        master.groupby(["primary_category", "primary_paradigm"])
        .agg(paper_count=("canonical_arxiv_id", "nunique"))
        .reset_index()
    )
    totals = stats.groupby("primary_category")["paper_count"].sum().reset_index().rename(columns={"paper_count": "category_total"})
    stats = stats.merge(totals, on="primary_category", how="left")
    stats["ratio_by_category"] = (stats["paper_count"] / stats["category_total"] * 100).round(2)
    stats = stats.sort_values(["primary_category", "paper_count"], ascending=[True, False])
    return stats


def create_confidence_stats(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if master.empty or "confidence" not in master.columns:
        empty_summary = pd.DataFrame(columns=["primary_paradigm", "paper_count", "mean", "median", "min", "max"])
        empty_bins = pd.DataFrame(columns=["confidence_bin", "paper_count"])
        return empty_summary, empty_bins

    working = master.copy()
    working["confidence"] = pd.to_numeric(working["confidence"], errors="coerce")

    summary = (
        working.groupby("primary_paradigm")
        .agg(
            paper_count=("canonical_arxiv_id", "nunique"),
            mean=("confidence", "mean"),
            median=("confidence", "median"),
            min=("confidence", "min"),
            max=("confidence", "max"),
        )
        .reset_index()
    )
    for col in ["mean", "median", "min", "max"]:
        summary[col] = summary[col].round(4)

    bins = pd.cut(
        working["confidence"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
        labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    )
    bin_df = (
        working.assign(confidence_bin=bins)
        .groupby("confidence_bin", observed=False)
        .agg(paper_count=("canonical_arxiv_id", "nunique"))
        .reset_index()
    )
    bin_df["confidence_bin"] = bin_df["confidence_bin"].astype(str)

    return summary.sort_values("paper_count", ascending=False), bin_df


def create_low_confidence_papers(master: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if master.empty or "confidence" not in master.columns:
        return pd.DataFrame()

    columns = [
        "canonical_arxiv_id",
        "title",
        "published_month",
        "primary_category",
        "primary_paradigm",
        "confidence",
        "contribution_type",
        "problem",
        "method",
        "result",
        "abs_url",
    ]
    columns = [col for col in columns if col in master.columns]
    df = master[pd.to_numeric(master["confidence"], errors="coerce") < threshold][columns].copy()
    return df.sort_values("confidence", ascending=True)


def create_keyword_stats(master: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if master.empty or "keywords" not in master.columns:
        return pd.DataFrame(columns=["keyword", "paper_count", "paradigms", "first_month", "latest_month", "rank"])

    rows = []
    for _, row in master.iterrows():
        arxiv_id = row.get("canonical_arxiv_id", "")
        month = row.get("published_month", "")
        paradigm = row.get("primary_paradigm", "Other")
        for keyword in split_list_field(row.get("keywords", "")):
            norm = normalize_keyword(keyword)
            if len(norm) < 2:
                continue
            rows.append(
                {
                    "canonical_arxiv_id": arxiv_id,
                    "published_month": month,
                    "primary_paradigm": paradigm,
                    "keyword": norm,
                }
            )

    keyword_df = pd.DataFrame(rows)
    if keyword_df.empty:
        return pd.DataFrame(columns=["keyword", "paper_count", "paradigms", "first_month", "latest_month", "rank"])

    stats = (
        keyword_df.groupby("keyword")
        .agg(
            paper_count=("canonical_arxiv_id", "nunique"),
            first_month=("published_month", "min"),
            latest_month=("published_month", "max"),
            paradigms=("primary_paradigm", lambda s: top_joined_values(s, limit=4)),
        )
        .reset_index()
        .sort_values("paper_count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    stats["rank"] = stats.index + 1
    return stats[["rank", "keyword", "paper_count", "paradigms", "first_month", "latest_month"]]


def create_concept_stats(concept_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if concept_df.empty:
        return pd.DataFrame(
            columns=["rank", "concept_name", "concept_type", "paper_count", "first_month", "latest_month", "top_paradigms", "top_categories"]
        )

    stats = (
        concept_df.groupby(["concept_name", "concept_type"])
        .agg(
            paper_count=("canonical_arxiv_id", "nunique"),
            first_month=("published_month", "min"),
            latest_month=("published_month", "max"),
            top_paradigms=("primary_paradigm", lambda s: top_joined_values(s, limit=4)),
            top_categories=("primary_category", lambda s: top_joined_values(s, limit=4)),
        )
        .reset_index()
        .sort_values("paper_count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    stats["rank"] = stats.index + 1
    return stats[["rank", "concept_name", "concept_type", "paper_count", "first_month", "latest_month", "top_paradigms", "top_categories"]]


def create_monthly_concept_stats(concept_df: pd.DataFrame) -> pd.DataFrame:
    if concept_df.empty:
        return pd.DataFrame(columns=["month", "concept_name", "concept_type", "paper_count", "rank_in_month"])

    stats = (
        concept_df.groupby(["published_month", "concept_name", "concept_type"])
        .agg(paper_count=("canonical_arxiv_id", "nunique"))
        .reset_index()
        .rename(columns={"published_month": "month"})
        .sort_values(["month", "paper_count"], ascending=[True, False])
    )
    stats["rank_in_month"] = stats.groupby("month")["paper_count"].rank(method="first", ascending=False).astype(int)
    return stats[stats["rank_in_month"] <= 50].copy()


def create_concept_type_stats(concept_df: pd.DataFrame) -> pd.DataFrame:
    if concept_df.empty:
        return pd.DataFrame(columns=["concept_type", "paper_count", "concept_count", "ratio"])

    stats = (
        concept_df.groupby("concept_type")
        .agg(
            paper_count=("canonical_arxiv_id", "nunique"),
            concept_count=("concept_name", "nunique"),
        )
        .reset_index()
        .sort_values("paper_count", ascending=False)
    )
    stats = add_ratio(stats)
    return stats


def create_concept_paradigm_matrix(concept_df: pd.DataFrame) -> pd.DataFrame:
    if concept_df.empty:
        return pd.DataFrame(columns=["concept_name", "concept_type", "primary_paradigm", "paper_count"])

    stats = (
        concept_df.groupby(["concept_name", "concept_type", "primary_paradigm"])
        .agg(paper_count=("canonical_arxiv_id", "nunique"))
        .reset_index()
        .sort_values("paper_count", ascending=False)
    )
    return stats


def create_concept_category_matrix(concept_df: pd.DataFrame) -> pd.DataFrame:
    if concept_df.empty:
        return pd.DataFrame(columns=["concept_name", "concept_type", "primary_category", "paper_count"])

    stats = (
        concept_df.groupby(["concept_name", "concept_type", "primary_category"])
        .agg(paper_count=("canonical_arxiv_id", "nunique"))
        .reset_index()
        .sort_values("paper_count", ascending=False)
    )
    return stats


def create_concept_edges(concept_df: pd.DataFrame, data: dict[str, pd.DataFrame], top_n: int) -> pd.DataFrame:
    explicit_edges = data.get("concept_edges", pd.DataFrame())

    if explicit_edges is not None and not explicit_edges.empty and {"source", "target"}.issubset(explicit_edges.columns):
        edges = explicit_edges.copy()
        if "edge_weight" not in edges.columns:
            edges["edge_weight"] = 1
        edges["edge_weight"] = pd.to_numeric(edges["edge_weight"], errors="coerce").fillna(1).astype(int)
        for col in ["first_month", "latest_month"]:
            if col not in edges.columns:
                edges[col] = ""
        return edges.sort_values("edge_weight", ascending=False).head(top_n).reset_index(drop=True)

    if concept_df.empty:
        return pd.DataFrame(columns=["source", "target", "edge_weight", "first_month", "latest_month"])

    rows = []
    for arxiv_id, group in concept_df.groupby("canonical_arxiv_id"):
        concepts = sorted(set(group["concept_name"].dropna().astype(str).tolist()))
        if len(concepts) < 2:
            continue
        months = group["published_month"].dropna().astype(str).tolist()
        month = months[0] if months else ""
        for source, target in itertools.combinations(concepts, 2):
            rows.append({"canonical_arxiv_id": arxiv_id, "source": source, "target": target, "published_month": month})

    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(columns=["source", "target", "edge_weight", "first_month", "latest_month"])

    edges = (
        raw.groupby(["source", "target"])
        .agg(
            edge_weight=("canonical_arxiv_id", "nunique"),
            first_month=("published_month", "min"),
            latest_month=("published_month", "max"),
        )
        .reset_index()
        .sort_values("edge_weight", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return edges


def create_representative_papers(master: pd.DataFrame, top_k_per_paradigm: int = 5) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()

    columns = [
        "canonical_arxiv_id",
        "title",
        "published_month",
        "primary_category",
        "primary_paradigm",
        "confidence",
        "contribution_type",
        "problem",
        "method",
        "result",
        "abs_url",
        "pdf_url",
    ]
    columns = [col for col in columns if col in master.columns]

    rows = []
    for paradigm, group in master.groupby("primary_paradigm"):
        group = group.copy()
        group["confidence"] = pd.to_numeric(group.get("confidence", 0), errors="coerce").fillna(0)
        selected = group.sort_values(["confidence", "published_month"], ascending=[False, False]).head(top_k_per_paradigm)
        rows.append(selected[columns])

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)


def create_insight_text_stats(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["field", "avg_chars", "median_chars", "empty_count"])

    rows = []
    for field in ["problem", "method", "result"]:
        if field not in master.columns:
            continue
        lengths = master[field].fillna("").astype(str).str.len()
        empty_count = int((lengths == 0).sum())
        rows.append(
            {
                "field": field,
                "avg_chars": round(float(lengths.mean()), 2),
                "median_chars": round(float(lengths.median()), 2),
                "empty_count": empty_count,
            }
        )
    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved {path} ({len(df)} rows)")


def save_run_log(
    output_dir: Path,
    llm_dir: Path,
    selected_months: list[str],
    row_counts: dict[str, int],
) -> None:
    log_path = output_dir / "llm_result_eda_run_log.csv"
    row = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "llm_dir": str(llm_dir),
        "output_dir": str(output_dir),
        "selected_months": " | ".join(selected_months),
    }
    row.update(row_counts)
    new_log = pd.DataFrame([row])

    if log_path.exists():
        old = pd.read_csv(log_path)
        new_log = pd.concat([old, new_log], ignore_index=True)

    save_dataframe(new_log, log_path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    llm_dir = Path(args.llm_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] LLM Result EDA started")
    print(f"[INFO] llm_dir: {llm_dir}")
    print(f"[INFO] output_dir: {output_dir}")

    data = load_llm_outputs(llm_dir)
    master = build_master_table(data)

    if master.empty:
        raise ValueError(
            "LLM 분석 결과에서 canonical_arxiv_id 기준 master table을 만들 수 없습니다. "
            "monthly_llm_eda.py 산출물이 생성되었는지 확인하세요."
        )

    concept_df = build_paper_concepts(data, master)

    master, concept_df, selected_months = filter_by_month(
        master=master,
        concept_df=concept_df,
        target_month=args.target_month,
        recent_months=args.recent_months,
    )

    print(f"[INFO] Selected months: {selected_months}")
    print(f"[INFO] Master rows after filter: {len(master)}")
    print(f"[INFO] Concept rows after filter: {len(concept_df)}")

    # Create derived EDA outputs.
    edge_df = create_concept_edges(concept_df, data, top_n=args.top_n_edges)
    kpi_summary = create_kpi_summary(master, concept_df, edge_df, selected_months, args.low_confidence_threshold)
    paradigm_stats = create_paradigm_stats(master)
    monthly_paradigm_stats = create_monthly_paradigm_stats(master)
    contribution_stats = create_contribution_stats(master)
    category_paradigm_matrix = create_category_paradigm_matrix(master)
    confidence_stats, confidence_bins = create_confidence_stats(master)
    low_confidence_papers = create_low_confidence_papers(master, threshold=args.low_confidence_threshold)
    keyword_stats = create_keyword_stats(master, top_n=args.top_n_keywords)
    concept_stats = create_concept_stats(concept_df, top_n=args.top_n_concepts)
    monthly_concept_stats = create_monthly_concept_stats(concept_df)
    concept_type_stats = create_concept_type_stats(concept_df)
    concept_paradigm_matrix = create_concept_paradigm_matrix(concept_df)
    concept_category_matrix = create_concept_category_matrix(concept_df)
    representative_papers = create_representative_papers(master)
    insight_text_stats = create_insight_text_stats(master)

    # Save normalized source-level tables as well, for Streamlit paper explorer.
    save_dataframe(master, output_dir / "llm_master_table.csv")
    save_dataframe(concept_df, output_dir / "llm_paper_concepts_enriched.csv")

    # Save aggregate tables.
    save_dataframe(kpi_summary, output_dir / "llm_kpi_summary.csv")
    save_dataframe(paradigm_stats, output_dir / "llm_paradigm_stats.csv")
    save_dataframe(monthly_paradigm_stats, output_dir / "llm_monthly_paradigm_stats.csv")
    save_dataframe(contribution_stats, output_dir / "llm_contribution_stats.csv")
    save_dataframe(category_paradigm_matrix, output_dir / "llm_category_paradigm_matrix.csv")
    save_dataframe(confidence_stats, output_dir / "llm_confidence_stats.csv")
    save_dataframe(confidence_bins, output_dir / "llm_confidence_bins.csv")
    save_dataframe(low_confidence_papers, output_dir / "llm_low_confidence_papers.csv")
    save_dataframe(keyword_stats, output_dir / "llm_keyword_stats.csv")
    save_dataframe(concept_stats, output_dir / "llm_concept_stats.csv")
    save_dataframe(monthly_concept_stats, output_dir / "llm_monthly_concept_stats.csv")
    save_dataframe(concept_type_stats, output_dir / "llm_concept_type_stats.csv")
    save_dataframe(concept_paradigm_matrix, output_dir / "llm_concept_paradigm_matrix.csv")
    save_dataframe(concept_category_matrix, output_dir / "llm_concept_category_matrix.csv")
    save_dataframe(edge_df, output_dir / "llm_concept_edges_ranked.csv")
    save_dataframe(representative_papers, output_dir / "llm_representative_papers.csv")
    save_dataframe(insight_text_stats, output_dir / "llm_insight_text_stats.csv")

    row_counts = {
        "master_rows": len(master),
        "concept_rows": len(concept_df),
        "edge_rows": len(edge_df),
        "paradigm_rows": len(paradigm_stats),
    }
    save_run_log(output_dir, llm_dir, selected_months, row_counts)

    print("[INFO] LLM Result EDA finished")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user")
        sys.exit(130)
