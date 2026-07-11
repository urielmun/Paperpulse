from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS


# -----------------------------------------------------------------------------
# PaperPulse AI - Daily Rule/Stat EDA Batch
# -----------------------------------------------------------------------------
# This script is intentionally independent of any LLM output.
# It reads only:
#   - data/processed/papers_clean.csv
#   - data/processed/paper_categories.csv
# and writes rule-based / statistical / probabilistic EDA outputs to:
#   - data/eda/
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_PAPERS_PATH = DEFAULT_PROCESSED_DIR / "papers_clean.csv"
DEFAULT_CATEGORIES_PATH = DEFAULT_PROCESSED_DIR / "paper_categories.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eda"

TARGET_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]

CUSTOM_STOPWORDS = set(ENGLISH_STOP_WORDS).union(
    {
        # Generic research terms
        "paper", "study", "studies", "method", "methods", "approach", "approaches",
        "result", "results", "show", "shows", "shown", "using", "based", "propose",
        "proposed", "present", "presents", "introduce", "introduced", "novel", "new",
        "task", "tasks", "dataset", "datasets", "data", "benchmark", "benchmarks",
        "evaluation", "experiment", "experiments", "performance", "state", "art",
        "state-of-the-art", "sota", "achieve", "achieves", "improve", "improves",
        "significant", "extensive", "various", "different", "multiple", "large", "scale",
        "language", "model", "models", "learning", "training", "neural", "network",
        "networks", "framework", "systems", "system", "work", "research", "problem",
        "problems", "paper", "analysis", "analyze", "demonstrate", "demonstrates",
        # arXiv / formatting artifacts
        "http", "https", "www", "org", "et", "al", "preprint", "arxiv",
        # Too broad AI terms
        "llm", "llms", "nlp", "ai", "ml",
    }
)

PARADIGM_RULES: dict[str, list[str]] = {
    "RAG/Retrieval": [
        r"\bretrieval\b", r"\bretriever\b", r"\bretrieve\b", r"\brag\b",
        r"retrieval[\- ]augmented", r"external knowledge", r"knowledge base",
        r"document grounding", r"grounded generation", r"open[\- ]book",
        r"dense passage", r"vector search", r"semantic search",
    ],
    "Agent/Tool Use": [
        r"\bagent\b", r"\bagents\b", r"tool use", r"tool[\- ]using",
        r"function calling", r"planning", r"plan generation", r"environment interaction",
        r"multi[\- ]agent", r"autonomous", r"workflow", r"actuation",
    ],
    "Alignment/Safety": [
        r"alignment", r"safety", r"safe", r"harmless", r"helpful", r"truthful",
        r"hallucination", r"hallucinations", r"toxicity", r"bias", r"fairness",
        r"jailbreak", r"red teaming", r"trustworthy", r"robustness", r"misinformation",
        r"preference optimization", r"rlhf", r"constitutional ai",
    ],
    "Reasoning": [
        r"reasoning", r"chain[\- ]of[\- ]thought", r"cot", r"mathematical reasoning",
        r"logical reasoning", r"multi[\- ]step", r"step[\- ]by[\- ]step", r"proof",
        r"planning reasoning", r"symbolic", r"deduction", r"induction",
    ],
    "Efficiency/Optimization": [
        r"efficient", r"efficiency", r"optimization", r"quantization", r"quantized",
        r"4[\- ]?bit", r"8[\- ]?bit", r"lora", r"qlora", r"adapter", r"adapters",
        r"distillation", r"pruning", r"sparsity", r"sparse", r"low[\- ]rank",
        r"peft", r"parameter[\- ]efficient", r"latency", r"throughput", r"memory",
        r"inference acceleration", r"speculative decoding", r"kv cache",
    ],
    "Multimodal": [
        r"multimodal", r"multi[\- ]modal", r"vision[\- ]language", r"vlm", r"image",
        r"images", r"video", r"audio", r"speech", r"visual", r"cross[\- ]modal",
        r"text[\- ]to[\- ]image", r"image[\- ]text",
    ],
    "Benchmark/Evaluation": [
        r"benchmark", r"benchmarks", r"evaluation", r"evaluate", r"leaderboard",
        r"test set", r"metrics", r"metric", r"human evaluation", r"automatic evaluation",
        r"dataset evaluation", r"comparative evaluation",
    ],
    "Dataset/Benchmark": [
        r"dataset", r"datasets", r"corpus", r"benchmark dataset", r"data collection",
        r"annotation", r"annotated", r"construct.*dataset", r"curated", r"databank",
    ],
    "Theory/Methodology": [
        r"theory", r"theoretical", r"formal", r"mathematical", r"principle",
        r"methodology", r"analysis of", r"understanding", r"mechanistic", r"interpretability",
        r"scaling law", r"generalization", r"convergence",
    ],
    "Application": [
        r"application", r"applications", r"medical", r"healthcare", r"legal", r"education",
        r"finance", r"robotics", r"science", r"biology", r"chemistry", r"code generation",
        r"software engineering", r"recommendation", r"dialogue system", r"question answering",
    ],
}

CURATION_KEYWORDS = {
    "rag": [r"\brag\b", r"retrieval[\- ]augmented", r"retrieval augmented generation"],
    "agent": [r"\bagent\b", r"\bagents\b", r"tool use", r"multi[\- ]agent"],
    "reasoning": [r"reasoning", r"chain[\- ]of[\- ]thought", r"multi[\- ]step"],
    "alignment": [r"alignment", r"rlhf", r"preference optimization", r"constitutional ai"],
    "safety": [r"safety", r"hallucination", r"toxicity", r"jailbreak", r"trustworthy"],
    "quantization": [r"quantization", r"quantized", r"4[\- ]?bit", r"8[\- ]?bit"],
    "lora": [r"\blora\b", r"\bqlora\b", r"low[\- ]rank", r"adapter"],
    "multimodal": [r"multimodal", r"multi[\- ]modal", r"vision[\- ]language", r"vlm"],
    "benchmark": [r"benchmark", r"leaderboard", r"evaluation", r"evaluate"],
    "hallucination": [r"hallucination", r"hallucinations", r"factuality", r"faithfulness"],
}

OPTIMIZATION_METHOD_RULES = {
    "LoRA": [r"\blora\b", r"low[\- ]rank adaptation"],
    "QLoRA": [r"\bqlora\b"],
    "Quantization": [r"quantization", r"quantized", r"4[\- ]?bit", r"8[\- ]?bit", r"int4", r"int8"],
    "Distillation": [r"distillation", r"distill", r"teacher[\- ]student"],
    "Pruning": [r"pruning", r"prune", r"pruned"],
    "Sparsity": [r"sparsity", r"sparse", r"mixture[\- ]of[\- ]experts", r"\bmoe\b"],
    "PEFT": [r"peft", r"parameter[\- ]efficient", r"adapter", r"adapters"],
    "Speculative Decoding": [r"speculative decoding", r"draft model"],
    "KV Cache Optimization": [r"kv cache", r"key[\- ]value cache"],
}

MODEL_SIZE_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(b|bn|billion|m|mn|million)\b",
    flags=re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily rule/stat/probabilistic EDA batch for PaperPulse AI"
    )

    parser.add_argument(
        "--papers-path",
        type=str,
        default=str(DEFAULT_PAPERS_PATH),
        help="Path to data/processed/papers_clean.csv",
    )
    parser.add_argument(
        "--categories-path",
        type=str,
        default=str(DEFAULT_CATEGORIES_PATH),
        help="Path to data/processed/paper_categories.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for EDA files",
    )
    parser.add_argument(
        "--target-month",
        type=str,
        default=None,
        help="Optional YYYY-MM filter. If omitted, analyze all months.",
    )
    parser.add_argument(
        "--top-n-keywords",
        type=int,
        default=80,
        help="Number of global top keywords to save",
    )
    parser.add_argument(
        "--top-n-per-month",
        type=int,
        default=50,
        help="Number of monthly keyword rows to retain per month",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for global vectorizers",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum features for keyword and topic vectorizers",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=8,
        help="Number of LDA topics",
    )
    parser.add_argument(
        "--topic-top-terms",
        type=int,
        default=12,
        help="Top terms per LDA topic",
    )
    parser.add_argument(
        "--skip-topic-model",
        action="store_true",
        help="Skip LDA topic modeling",
    )
    parser.add_argument(
        "--min-rising-count",
        type=int,
        default=2,
        help="Minimum monthly keyword count for rising keyword output",
    )

    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {path} ({len(df)} rows)")


def normalize_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def extract_categories_from_text(categories_text: Any) -> list[str]:
    if pd.isna(categories_text):
        return []
    return [cat.strip() for cat in str(categories_text).split(",") if cat.strip()]


def compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def count_pattern_matches(text: str, patterns: list[str]) -> int:
    count = 0
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            count += 1
    return count


def contains_any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def get_effective_min_df(n_docs: int, requested_min_df: int) -> int:
    if n_docs <= 1:
        return 1
    return max(1, min(requested_min_df, n_docs))


def get_effective_max_df(n_docs: int) -> float:
    # max_df=0.85 can be too aggressive for tiny monthly subsets.
    if n_docs < 5:
        return 1.0
    return 0.90


def make_count_vectorizer(
    n_docs: int,
    min_df: int,
    max_features: int,
    ngram_range: tuple[int, int] = (1, 2),
) -> CountVectorizer:
    return CountVectorizer(
        stop_words=list(CUSTOM_STOPWORDS),
        ngram_range=ngram_range,
        min_df=get_effective_min_df(n_docs, min_df),
        max_df=get_effective_max_df(n_docs),
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
    )


def make_tfidf_vectorizer(
    n_docs: int,
    min_df: int,
    max_features: int,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words=list(CUSTOM_STOPWORDS),
        ngram_range=ngram_range,
        min_df=get_effective_min_df(n_docs, min_df),
        max_df=get_effective_max_df(n_docs),
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_processed_data(
    papers_path: Path,
    categories_path: Path,
    target_month: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not papers_path.exists():
        raise FileNotFoundError(
            f"papers_clean.csv not found: {papers_path}\n"
            "먼저 preprocess_papers.py를 실행해야 합니다."
        )

    papers_df = pd.read_csv(papers_path, dtype={"canonical_arxiv_id": str})

    if papers_df.empty:
        raise ValueError("papers_clean.csv is empty.")

    required_cols = [
        "canonical_arxiv_id",
        "title",
        "summary",
        "clean_title",
        "clean_abstract",
        "authors_text",
        "primary_category",
        "categories_text",
        "published_at",
        "updated_at",
        "abs_url",
        "pdf_url",
    ]
    missing = [col for col in required_cols if col not in papers_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in papers_clean.csv: {missing}")

    papers_df["canonical_arxiv_id"] = papers_df["canonical_arxiv_id"].astype(str).str.strip()
    papers_df = papers_df[papers_df["canonical_arxiv_id"] != ""].copy()

    papers_df["published_at"] = safe_to_datetime(papers_df["published_at"])
    papers_df["updated_at"] = safe_to_datetime(papers_df["updated_at"])
    papers_df["published_month"] = papers_df["published_at"].dt.to_period("M").astype(str)
    papers_df["published_year"] = papers_df["published_at"].dt.year
    papers_df["analysis_text"] = (
        papers_df["clean_title"].fillna("") + " " + papers_df["clean_abstract"].fillna("")
    ).apply(normalize_text)

    # Fallback if clean columns are unexpectedly empty.
    empty_text_mask = papers_df["analysis_text"].str.len() == 0
    if empty_text_mask.any():
        papers_df.loc[empty_text_mask, "analysis_text"] = (
            papers_df.loc[empty_text_mask, "title"].fillna("")
            + " "
            + papers_df.loc[empty_text_mask, "summary"].fillna("")
        ).apply(normalize_text)

    if target_month:
        if not re.fullmatch(r"\d{4}-\d{2}", target_month):
            raise ValueError("--target-month must be YYYY-MM")
        papers_df = papers_df[papers_df["published_month"] == target_month].copy()

    if categories_path.exists():
        categories_df = pd.read_csv(categories_path, dtype={"canonical_arxiv_id": str})
        if not categories_df.empty:
            categories_df["canonical_arxiv_id"] = categories_df["canonical_arxiv_id"].astype(str).str.strip()
            categories_df["category"] = categories_df["category"].astype(str).str.strip()
    else:
        rows = []
        for _, row in papers_df.iterrows():
            for category in extract_categories_from_text(row.get("categories_text", "")):
                rows.append({"canonical_arxiv_id": row["canonical_arxiv_id"], "category": category})
        categories_df = pd.DataFrame(rows)

    if target_month and not categories_df.empty:
        categories_df = categories_df.merge(
            papers_df[["canonical_arxiv_id"]],
            on="canonical_arxiv_id",
            how="inner",
        )

    categories_df = categories_df.drop_duplicates(
        subset=["canonical_arxiv_id", "category"]
    ).copy()

    return papers_df.reset_index(drop=True), categories_df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Basic statistical EDA
# -----------------------------------------------------------------------------


def create_monthly_stats(papers_df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        papers_df.dropna(subset=["published_month"])
        .groupby("published_month")
        .agg(
            paper_count=("canonical_arxiv_id", "nunique"),
            unique_primary_categories=("primary_category", "nunique"),
        )
        .reset_index()
        .rename(columns={"published_month": "month"})
        .sort_values("month")
    )

    monthly["previous_month_count"] = monthly["paper_count"].shift(1).fillna(0).astype(int)
    denominator = monthly["previous_month_count"].replace(0, np.nan)
    monthly["month_over_month_growth_rate"] = (
        (monthly["paper_count"] - monthly["previous_month_count"]) / denominator * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0).round(2)

    monthly["rolling_3month_avg"] = monthly["paper_count"].rolling(3, min_periods=1).mean().round(2)

    return monthly


def create_category_stats(categories_df: pd.DataFrame) -> pd.DataFrame:
    if categories_df.empty:
        return pd.DataFrame(columns=["category", "paper_count", "category_ratio"])

    total_papers = categories_df["canonical_arxiv_id"].nunique()

    category_stats = (
        categories_df.groupby("category")
        .agg(paper_count=("canonical_arxiv_id", "nunique"))
        .reset_index()
        .sort_values("paper_count", ascending=False)
    )

    category_stats["category_ratio"] = (
        category_stats["paper_count"] / max(total_papers, 1) * 100
    ).round(2)

    return category_stats


def create_monthly_category_stats(
    papers_df: pd.DataFrame,
    categories_df: pd.DataFrame,
) -> pd.DataFrame:
    if categories_df.empty:
        return pd.DataFrame(columns=["month", "category", "paper_count", "month_total", "ratio"])

    merged = categories_df.merge(
        papers_df[["canonical_arxiv_id", "published_month"]],
        on="canonical_arxiv_id",
        how="left",
    )

    stats = (
        merged.dropna(subset=["published_month", "category"])
        .groupby(["published_month", "category"])
        .agg(paper_count=("canonical_arxiv_id", "nunique"))
        .reset_index()
        .rename(columns={"published_month": "month"})
        .sort_values(["month", "paper_count"], ascending=[True, False])
    )

    if stats.empty:
        return stats

    totals = stats.groupby("month")["paper_count"].sum().reset_index().rename(
        columns={"paper_count": "month_total"}
    )
    stats = stats.merge(totals, on="month", how="left")
    stats["ratio"] = (stats["paper_count"] / stats["month_total"] * 100).round(2)

    return stats


def create_kpi_summary(
    papers_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    monthly_stats: pd.DataFrame,
    category_stats: pd.DataFrame,
    top_keywords: pd.DataFrame,
    rising_keywords: pd.DataFrame,
) -> pd.DataFrame:
    total_papers = papers_df["canonical_arxiv_id"].nunique()
    total_categories = categories_df["category"].nunique() if not categories_df.empty else 0
    first_month = papers_df["published_month"].min() if not papers_df.empty else ""
    latest_month = papers_df["published_month"].max() if not papers_df.empty else ""

    latest_month_count = 0
    latest_growth = 0.0
    if not monthly_stats.empty:
        latest_row = monthly_stats.sort_values("month").iloc[-1]
        latest_month_count = int(latest_row["paper_count"])
        latest_growth = float(latest_row["month_over_month_growth_rate"])

    top_category = ""
    top_category_count = 0
    if not category_stats.empty:
        top_category = str(category_stats.iloc[0]["category"])
        top_category_count = int(category_stats.iloc[0]["paper_count"])

    top_keyword = ""
    top_keyword_count = 0
    if not top_keywords.empty:
        top_keyword = str(top_keywords.iloc[0]["keyword"])
        top_keyword_count = int(top_keywords.iloc[0]["keyword_count"])

    top_rising_keyword = ""
    top_rising_score = 0.0
    if not rising_keywords.empty:
        latest_rising = rising_keywords[rising_keywords["month"] == rising_keywords["month"].max()]
        if not latest_rising.empty:
            row = latest_rising.sort_values("trend_score", ascending=False).iloc[0]
            top_rising_keyword = str(row["keyword"])
            top_rising_score = float(row["trend_score"])

    rows = [
        ("total_papers", total_papers, "전체 분석 논문 수"),
        ("total_categories", total_categories, "전체 카테고리 수"),
        ("first_month", first_month, "가장 오래된 논문 월"),
        ("latest_month", latest_month, "가장 최근 논문 월"),
        ("latest_month_paper_count", latest_month_count, "가장 최근 월 논문 수"),
        ("latest_month_growth_rate", latest_growth, "가장 최근 월 전월 대비 증가율"),
        ("top_category", top_category, "최다 카테고리"),
        ("top_category_count", top_category_count, "최다 카테고리 논문 수"),
        ("top_keyword", top_keyword, "전체 데이터 기준 최상위 키워드"),
        ("top_keyword_count", top_keyword_count, "최상위 키워드 등장 횟수"),
        ("top_rising_keyword", top_rising_keyword, "최근 월 기준 급상승 키워드"),
        ("top_rising_score", round(top_rising_score, 4), "최근 월 기준 급상승 점수"),
    ]

    return pd.DataFrame(
        [
            {"metric_name": name, "metric_value": value, "metric_description": desc}
            for name, value, desc in rows
        ]
    )


# -----------------------------------------------------------------------------
# Keyword EDA
# -----------------------------------------------------------------------------


def create_top_keywords(
    papers_df: pd.DataFrame,
    top_n: int,
    min_df: int,
    max_features: int,
) -> pd.DataFrame:
    texts = papers_df["analysis_text"].fillna("").tolist()
    n_docs = len(texts)

    if n_docs == 0:
        return pd.DataFrame(columns=["rank", "keyword", "keyword_count", "tfidf_score"])

    try:
        count_vectorizer = make_count_vectorizer(n_docs, min_df, max_features)
        X_count = count_vectorizer.fit_transform(texts)
        terms = count_vectorizer.get_feature_names_out()
        counts = X_count.sum(axis=0).A1

        tfidf_vectorizer = make_tfidf_vectorizer(n_docs, min_df, max_features)
        X_tfidf = tfidf_vectorizer.fit_transform(texts)
        tfidf_terms = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = X_tfidf.sum(axis=0).A1
        tfidf_map = dict(zip(tfidf_terms, tfidf_scores))

        df = pd.DataFrame(
            {
                "keyword": terms,
                "keyword_count": counts,
                "tfidf_score": [round(float(tfidf_map.get(term, 0.0)), 6) for term in terms],
            }
        )

        df = df.sort_values(["keyword_count", "tfidf_score"], ascending=[False, False]).head(top_n)
        df = df.reset_index(drop=True)
        df["rank"] = df.index + 1
        return df[["rank", "keyword", "keyword_count", "tfidf_score"]]

    except ValueError as e:
        print(f"[WARN] Top keyword extraction skipped: {e}")
        return pd.DataFrame(columns=["rank", "keyword", "keyword_count", "tfidf_score"])


def create_keyword_month_matrix(
    papers_df: pd.DataFrame,
    min_df: int,
    max_features: int,
) -> tuple[pd.DataFrame, CountVectorizer | None]:
    texts = papers_df["analysis_text"].fillna("").tolist()
    n_docs = len(texts)

    if n_docs == 0:
        return pd.DataFrame(columns=["month", "keyword", "keyword_count"]), None

    try:
        vectorizer = make_count_vectorizer(n_docs, min_df, max_features)
        X = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()
    except ValueError as e:
        print(f"[WARN] Keyword matrix skipped: {e}")
        return pd.DataFrame(columns=["month", "keyword", "keyword_count"]), None

    months = sorted(papers_df["published_month"].dropna().unique().tolist())
    rows = []

    for month in months:
        idx = papers_df.index[papers_df["published_month"] == month].tolist()
        if not idx:
            continue
        counts = X[idx].sum(axis=0).A1
        for term, count in zip(terms, counts):
            rows.append({"month": month, "keyword": term, "keyword_count": int(count)})

    matrix_df = pd.DataFrame(rows)

    return matrix_df, vectorizer


def create_keyword_stats_and_rising(
    papers_df: pd.DataFrame,
    top_n_per_month: int,
    min_df: int,
    max_features: int,
    min_rising_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_df, _ = create_keyword_month_matrix(papers_df, min_df, max_features)

    if matrix_df.empty:
        empty_stats = pd.DataFrame(
            columns=[
                "month", "keyword", "keyword_count", "previous_month_count",
                "growth_rate", "trend_score", "rank_in_month",
            ]
        )
        return empty_stats, empty_stats.copy()

    matrix_df = matrix_df.sort_values(["keyword", "month"])
    matrix_df["previous_month_count"] = matrix_df.groupby("keyword")["keyword_count"].shift(1).fillna(0).astype(int)

    denominator = matrix_df["previous_month_count"].where(matrix_df["previous_month_count"] > 0, 1)
    growth_ratio = (matrix_df["keyword_count"] - matrix_df["previous_month_count"]) / denominator
    matrix_df["growth_rate"] = (growth_ratio * 100).round(2)

    # A stability-aware rising score. Frequency and growth must both matter.
    matrix_df["trend_score"] = (
        matrix_df["growth_rate"] * np.log1p(matrix_df["keyword_count"])
    ).round(4)

    matrix_df = matrix_df[matrix_df["keyword_count"] > 0].copy()
    matrix_df = matrix_df.sort_values(["month", "keyword_count", "trend_score"], ascending=[True, False, False])
    matrix_df["rank_in_month"] = matrix_df.groupby("month")["keyword_count"].rank(method="first", ascending=False).astype(int)

    keyword_stats = matrix_df[matrix_df["rank_in_month"] <= top_n_per_month].copy()

    rising = matrix_df[
        (matrix_df["keyword_count"] >= min_rising_count)
        & (matrix_df["trend_score"] > 0)
    ].copy()
    rising = rising.sort_values(["month", "trend_score"], ascending=[True, False])
    rising["rising_rank_in_month"] = rising.groupby("month")["trend_score"].rank(method="first", ascending=False).astype(int)
    rising = rising[rising["rising_rank_in_month"] <= top_n_per_month].copy()

    return keyword_stats.reset_index(drop=True), rising.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Category synergy EDA
# -----------------------------------------------------------------------------


def create_category_synergy_stats(
    papers_df: pd.DataFrame,
    categories_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if categories_df.empty:
        empty_combo = pd.DataFrame(columns=["category_combo", "combo_size", "paper_count", "paper_ratio"])
        empty_month = pd.DataFrame(columns=["month", "category_combo", "combo_size", "paper_count", "month_total", "ratio"])
        empty_pair = pd.DataFrame(columns=["source", "target", "paper_count", "paper_ratio"])
        empty_pair_month = pd.DataFrame(columns=["month", "source", "target", "paper_count"])
        return empty_combo, empty_month, empty_pair, empty_pair_month

    merged = categories_df.merge(
        papers_df[["canonical_arxiv_id", "published_month"]],
        on="canonical_arxiv_id",
        how="inner",
    )

    total_papers = papers_df["canonical_arxiv_id"].nunique()
    combo_counter = Counter()
    monthly_combo_counter = Counter()
    pair_counter = Counter()
    monthly_pair_counter = Counter()

    for arxiv_id, group in merged.groupby("canonical_arxiv_id"):
        categories = sorted(set(group["category"].dropna().astype(str).tolist()))
        # Focus on the planned categories, but keep all if the paper has none of the target cats.
        target_cats = [cat for cat in categories if cat in TARGET_CATEGORIES]
        selected = target_cats if target_cats else categories
        selected = sorted(set(selected))

        if not selected:
            continue

        month = str(group["published_month"].dropna().iloc[0]) if not group["published_month"].dropna().empty else ""
        combo = " + ".join(selected)
        combo_counter[(combo, len(selected))] += 1
        monthly_combo_counter[(month, combo, len(selected))] += 1

        for source, target in itertools.combinations(selected, 2):
            pair_counter[(source, target)] += 1
            monthly_pair_counter[(month, source, target)] += 1

    combo_rows = [
        {
            "category_combo": combo,
            "combo_size": combo_size,
            "paper_count": count,
            "paper_ratio": round(count / max(total_papers, 1) * 100, 2),
        }
        for (combo, combo_size), count in combo_counter.items()
    ]
    synergy_df = pd.DataFrame(combo_rows).sort_values("paper_count", ascending=False) if combo_rows else pd.DataFrame(
        columns=["category_combo", "combo_size", "paper_count", "paper_ratio"]
    )

    monthly_rows = [
        {
            "month": month,
            "category_combo": combo,
            "combo_size": combo_size,
            "paper_count": count,
        }
        for (month, combo, combo_size), count in monthly_combo_counter.items()
    ]
    monthly_df = pd.DataFrame(monthly_rows)
    if not monthly_df.empty:
        totals = monthly_df.groupby("month")["paper_count"].sum().reset_index().rename(columns={"paper_count": "month_total"})
        monthly_df = monthly_df.merge(totals, on="month", how="left")
        monthly_df["ratio"] = (monthly_df["paper_count"] / monthly_df["month_total"] * 100).round(2)
        monthly_df = monthly_df.sort_values(["month", "paper_count"], ascending=[True, False])
    else:
        monthly_df = pd.DataFrame(columns=["month", "category_combo", "combo_size", "paper_count", "month_total", "ratio"])

    pair_rows = [
        {
            "source": source,
            "target": target,
            "paper_count": count,
            "paper_ratio": round(count / max(total_papers, 1) * 100, 2),
        }
        for (source, target), count in pair_counter.items()
    ]
    pair_df = pd.DataFrame(pair_rows).sort_values("paper_count", ascending=False) if pair_rows else pd.DataFrame(
        columns=["source", "target", "paper_count", "paper_ratio"]
    )

    monthly_pair_rows = [
        {"month": month, "source": source, "target": target, "paper_count": count}
        for (month, source, target), count in monthly_pair_counter.items()
    ]
    monthly_pair_df = pd.DataFrame(monthly_pair_rows).sort_values(["month", "paper_count"], ascending=[True, False]) if monthly_pair_rows else pd.DataFrame(
        columns=["month", "source", "target", "paper_count"]
    )

    return synergy_df, monthly_df, pair_df, monthly_pair_df


# -----------------------------------------------------------------------------
# Rule-based research paradigm EDA
# -----------------------------------------------------------------------------


def classify_rule_paradigm(text: str) -> tuple[str, list[str], dict[str, int], int]:
    scores = {label: count_pattern_matches(text, patterns) for label, patterns in PARADIGM_RULES.items()}
    positive = {label: score for label, score in scores.items() if score > 0}

    if not positive:
        return "Other", [], scores, 0

    primary = sorted(positive.items(), key=lambda x: (-x[1], x[0]))[0][0]
    secondary = [label for label, score in sorted(positive.items(), key=lambda x: (-x[1], x[0])) if label != primary][:3]
    total_score = sum(positive.values())

    return primary, secondary, scores, total_score


def create_rule_paradigm_labels(papers_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for _, row in papers_df.iterrows():
        primary, secondary, scores, total_score = classify_rule_paradigm(row["analysis_text"])
        confidence_proxy = 0.0
        if total_score > 0:
            confidence_proxy = round(scores.get(primary, 0) / total_score, 4)

        rows.append(
            {
                "canonical_arxiv_id": row["canonical_arxiv_id"],
                "published_month": row["published_month"],
                "primary_category": row.get("primary_category", ""),
                "categories_text": row.get("categories_text", ""),
                "rule_primary_paradigm": primary,
                "rule_secondary_paradigms": " | ".join(secondary),
                "rule_confidence_proxy": confidence_proxy,
                "rule_total_score": total_score,
                "rule_score_json": json.dumps(scores, ensure_ascii=False),
            }
        )

    label_df = pd.DataFrame(rows)

    if label_df.empty:
        stats_df = pd.DataFrame(columns=["month", "rule_primary_paradigm", "paper_count", "month_total", "ratio"])
        return label_df, stats_df

    stats_df = (
        label_df.groupby(["published_month", "rule_primary_paradigm"])
        .agg(paper_count=("canonical_arxiv_id", "nunique"))
        .reset_index()
        .rename(columns={"published_month": "month"})
    )

    totals = stats_df.groupby("month")["paper_count"].sum().reset_index().rename(columns={"paper_count": "month_total"})
    stats_df = stats_df.merge(totals, on="month", how="left")
    stats_df["ratio"] = (stats_df["paper_count"] / stats_df["month_total"] * 100).round(2)
    stats_df = stats_df.sort_values(["month", "paper_count"], ascending=[True, False])

    return label_df, stats_df


# -----------------------------------------------------------------------------
# Model efficiency EDA
# -----------------------------------------------------------------------------


def extract_model_sizes(text: str) -> list[tuple[str, float]]:
    sizes = []
    for match in MODEL_SIZE_PATTERN.finditer(text):
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit in {"m", "mn", "million"}:
            value_billion = value / 1000.0
            label = f"{value:g}M"
        else:
            value_billion = value
            label = f"{value:g}B"
        sizes.append((label, round(value_billion, 4)))

    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for label, value in sizes:
        key = (label, value)
        if key not in seen:
            seen.add(key)
            unique.append((label, value))
    return unique


def extract_optimization_methods(text: str) -> list[str]:
    methods = []
    for method, patterns in OPTIMIZATION_METHOD_RULES.items():
        if contains_any_pattern(text, patterns):
            methods.append(method)
    return methods


def create_model_efficiency_stats(papers_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paper_rows = []

    for _, row in papers_df.iterrows():
        text = row["analysis_text"]
        sizes = extract_model_sizes(text)
        methods = extract_optimization_methods(text)

        if not sizes and not methods:
            continue

        if not sizes:
            sizes = [("not_mentioned", np.nan)]
        if not methods:
            methods = ["not_mentioned"]

        for label, size_billion in sizes:
            for method in methods:
                paper_rows.append(
                    {
                        "canonical_arxiv_id": row["canonical_arxiv_id"],
                        "published_month": row["published_month"],
                        "primary_category": row.get("primary_category", ""),
                        "model_size_label": label,
                        "model_size_billion": size_billion,
                        "optimization_method": method,
                    }
                )

    paper_df = pd.DataFrame(paper_rows)

    if paper_df.empty:
        empty_paper = pd.DataFrame(
            columns=[
                "canonical_arxiv_id", "published_month", "primary_category",
                "model_size_label", "model_size_billion", "optimization_method",
            ]
        )
        empty_stats = pd.DataFrame(
            columns=["month", "model_size_label", "model_size_billion", "optimization_method", "paper_count"]
        )
        return empty_paper, empty_stats

    stats = (
        paper_df.groupby(["published_month", "model_size_label", "model_size_billion", "optimization_method"], dropna=False)
        .agg(paper_count=("canonical_arxiv_id", "nunique"))
        .reset_index()
        .rename(columns={"published_month": "month"})
        .sort_values(["month", "paper_count"], ascending=[True, False])
    )

    return paper_df, stats


# -----------------------------------------------------------------------------
# Curation and co-occurrence graph
# -----------------------------------------------------------------------------


def create_curation_outputs(
    papers_df: pd.DataFrame,
    min_df: int,
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keyword_rows = []
    related_rows = []
    recent_rows = []

    for curation_keyword, patterns in CURATION_KEYWORDS.items():
        mask = papers_df["analysis_text"].apply(lambda text: contains_any_pattern(text, patterns))
        subset = papers_df[mask].copy()

        if subset.empty:
            keyword_rows.append(
                {
                    "curation_keyword": curation_keyword,
                    "paper_count": 0,
                    "first_month": "",
                    "latest_month": "",
                    "top_category": "",
                    "latest_month_paper_count": 0,
                }
            )
            continue

        top_category = subset["primary_category"].fillna("").value_counts().index[0]
        latest_month = subset["published_month"].max()
        latest_count = int((subset["published_month"] == latest_month).sum())

        keyword_rows.append(
            {
                "curation_keyword": curation_keyword,
                "paper_count": subset["canonical_arxiv_id"].nunique(),
                "first_month": subset["published_month"].min(),
                "latest_month": latest_month,
                "top_category": top_category,
                "latest_month_paper_count": latest_count,
            }
        )

        # Related terms inside this keyword neighborhood.
        texts = subset["analysis_text"].fillna("").tolist()
        try:
            vectorizer = make_count_vectorizer(
                n_docs=len(texts),
                min_df=1,
                max_features=max_features,
                ngram_range=(1, 2),
            )
            X = vectorizer.fit_transform(texts)
            terms = vectorizer.get_feature_names_out()
            counts = X.sum(axis=0).A1

            related_df = pd.DataFrame({"related_keyword": terms, "keyword_count": counts})
            # Remove the curation keyword itself and very direct aliases.
            related_df = related_df[
                ~related_df["related_keyword"].str.contains(curation_keyword, regex=False)
            ].copy()
            related_df = related_df.sort_values("keyword_count", ascending=False).head(20)

            for rank, (_, r) in enumerate(related_df.iterrows(), start=1):
                related_rows.append(
                    {
                        "curation_keyword": curation_keyword,
                        "rank": rank,
                        "related_keyword": r["related_keyword"],
                        "keyword_count": int(r["keyword_count"]),
                    }
                )
        except ValueError:
            pass

        recent_subset = subset.sort_values("published_at", ascending=False).head(20)
        for _, p in recent_subset.iterrows():
            recent_rows.append(
                {
                    "curation_keyword": curation_keyword,
                    "canonical_arxiv_id": p["canonical_arxiv_id"],
                    "title": p.get("title", ""),
                    "published_month": p.get("published_month", ""),
                    "primary_category": p.get("primary_category", ""),
                    "abs_url": p.get("abs_url", ""),
                    "pdf_url": p.get("pdf_url", ""),
                }
            )

    return (
        pd.DataFrame(keyword_rows).sort_values("paper_count", ascending=False),
        pd.DataFrame(related_rows),
        pd.DataFrame(recent_rows),
    )


def create_keyword_cooccurrence_graph(
    papers_df: pd.DataFrame,
    top_keywords: pd.DataFrame,
    max_keywords: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if top_keywords.empty:
        empty_nodes = pd.DataFrame(columns=["keyword", "paper_count", "keyword_count"])
        empty_edges = pd.DataFrame(columns=["source", "target", "edge_weight"])
        return empty_nodes, empty_edges

    keywords = top_keywords.head(max_keywords)["keyword"].dropna().astype(str).tolist()
    keyword_patterns = {
        kw: re.compile(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", flags=re.IGNORECASE)
        for kw in keywords
    }

    node_paper_counter = Counter()
    edge_counter = Counter()

    for _, row in papers_df.iterrows():
        text = row["analysis_text"]
        found = sorted([kw for kw, pattern in keyword_patterns.items() if pattern.search(text)])

        for kw in found:
            node_paper_counter[kw] += 1

        for source, target in itertools.combinations(found, 2):
            edge_counter[(source, target)] += 1

    count_map = dict(zip(top_keywords["keyword"], top_keywords["keyword_count"]))

    nodes = pd.DataFrame(
        [
            {
                "keyword": kw,
                "paper_count": node_paper_counter.get(kw, 0),
                "keyword_count": int(count_map.get(kw, 0)),
            }
            for kw in keywords
            if node_paper_counter.get(kw, 0) > 0
        ]
    ).sort_values("paper_count", ascending=False) if keywords else pd.DataFrame(
        columns=["keyword", "paper_count", "keyword_count"]
    )

    edges = pd.DataFrame(
        [
            {"source": source, "target": target, "edge_weight": count}
            for (source, target), count in edge_counter.items()
        ]
    ).sort_values("edge_weight", ascending=False) if edge_counter else pd.DataFrame(
        columns=["source", "target", "edge_weight"]
    )

    return nodes, edges


# -----------------------------------------------------------------------------
# Probabilistic topic modeling using LDA
# -----------------------------------------------------------------------------


def create_lda_topic_outputs(
    papers_df: pd.DataFrame,
    n_topics: int,
    top_terms: int,
    min_df: int,
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_docs = len(papers_df)
    empty_terms = pd.DataFrame(columns=["topic_id", "rank", "term", "weight"])
    empty_paper = pd.DataFrame(columns=["canonical_arxiv_id", "published_month", "dominant_topic", "topic_probability"])
    empty_monthly = pd.DataFrame(columns=["month", "topic_id", "paper_count", "month_total", "ratio", "avg_probability"])
    empty_labels = pd.DataFrame(columns=["topic_id", "topic_label", "top_terms"])

    if n_docs < 2:
        print("[WARN] LDA skipped: not enough documents")
        return empty_terms, empty_paper, empty_monthly, empty_labels

    effective_topics = max(2, min(n_topics, n_docs))

    texts = papers_df["analysis_text"].fillna("").tolist()

    try:
        vectorizer = make_count_vectorizer(
            n_docs=n_docs,
            min_df=min_df,
            max_features=max_features,
            ngram_range=(1, 2),
        )
        X = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()
    except ValueError as e:
        print(f"[WARN] LDA skipped at vectorizer stage: {e}")
        return empty_terms, empty_paper, empty_monthly, empty_labels

    if X.shape[1] < 2:
        print("[WARN] LDA skipped: not enough vocabulary")
        return empty_terms, empty_paper, empty_monthly, empty_labels

    effective_topics = min(effective_topics, X.shape[0], max(2, X.shape[1] // 2))

    lda = LatentDirichletAllocation(
        n_components=effective_topics,
        random_state=42,
        learning_method="batch",
        max_iter=20,
        evaluate_every=-1,
    )

    topic_probs = lda.fit_transform(X)
    dominant_topic = topic_probs.argmax(axis=1)
    dominant_prob = topic_probs.max(axis=1)

    term_rows = []
    label_rows = []

    for topic_id, topic_weights in enumerate(lda.components_):
        top_indices = topic_weights.argsort()[::-1][:top_terms]
        top_term_list = []
        for rank, term_idx in enumerate(top_indices, start=1):
            term = terms[term_idx]
            weight = float(topic_weights[term_idx])
            top_term_list.append(term)
            term_rows.append(
                {
                    "topic_id": topic_id,
                    "rank": rank,
                    "term": term,
                    "weight": round(weight, 6),
                }
            )
        label_rows.append(
            {
                "topic_id": topic_id,
                "topic_label": " / ".join(top_term_list[:4]),
                "top_terms": " | ".join(top_term_list),
            }
        )

    paper_topic_df = pd.DataFrame(
        {
            "canonical_arxiv_id": papers_df["canonical_arxiv_id"].values,
            "published_month": papers_df["published_month"].values,
            "dominant_topic": dominant_topic,
            "topic_probability": np.round(dominant_prob, 6),
        }
    )

    monthly = (
        paper_topic_df.groupby(["published_month", "dominant_topic"])
        .agg(
            paper_count=("canonical_arxiv_id", "nunique"),
            avg_probability=("topic_probability", "mean"),
        )
        .reset_index()
        .rename(columns={"published_month": "month", "dominant_topic": "topic_id"})
    )

    totals = monthly.groupby("month")["paper_count"].sum().reset_index().rename(columns={"paper_count": "month_total"})
    monthly = monthly.merge(totals, on="month", how="left")
    monthly["ratio"] = (monthly["paper_count"] / monthly["month_total"] * 100).round(2)
    monthly["avg_probability"] = monthly["avg_probability"].round(6)
    monthly = monthly.sort_values(["month", "paper_count"], ascending=[True, False])

    return (
        pd.DataFrame(term_rows),
        paper_topic_df,
        monthly,
        pd.DataFrame(label_rows),
    )


# -----------------------------------------------------------------------------
# Run log and orchestration
# -----------------------------------------------------------------------------


def save_run_log(
    output_dir: Path,
    papers_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    log_path = output_dir / "daily_eda_run_log.csv"
    row = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "papers_path": str(Path(args.papers_path).resolve()),
        "categories_path": str(Path(args.categories_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "target_month": args.target_month or "ALL",
        "paper_count": papers_df["canonical_arxiv_id"].nunique(),
        "paper_category_rows": len(categories_df),
        "min_month": papers_df["published_month"].min() if not papers_df.empty else "",
        "max_month": papers_df["published_month"].max() if not papers_df.empty else "",
        "top_n_keywords": args.top_n_keywords,
        "top_n_per_month": args.top_n_per_month,
        "min_df": args.min_df,
        "max_features": args.max_features,
        "n_topics": args.n_topics,
        "skip_topic_model": args.skip_topic_model,
    }

    log_df = pd.DataFrame([row])
    if log_path.exists():
        old_df = pd.read_csv(log_path)
        log_df = pd.concat([old_df, log_df], ignore_index=True)
    save_csv(log_df, log_path)


def run_daily_rule_stat_eda(args: argparse.Namespace) -> None:
    papers_path = Path(args.papers_path).resolve()
    categories_path = Path(args.categories_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_output_dir(output_dir)

    print("[INFO] Daily rule/stat/probabilistic EDA started")
    print(f"[INFO] Papers path: {papers_path}")
    print(f"[INFO] Categories path: {categories_path}")
    print(f"[INFO] Output dir: {output_dir}")
    print("[INFO] LLM outputs are not read by this script.")

    papers_df, categories_df = load_processed_data(
        papers_path=papers_path,
        categories_path=categories_path,
        target_month=args.target_month,
    )

    print(f"[INFO] Loaded papers: {len(papers_df)}")
    print(f"[INFO] Loaded paper-category rows: {len(categories_df)}")

    if papers_df.empty:
        raise ValueError("No papers to analyze after filtering.")

    # Basic stats
    monthly_stats = create_monthly_stats(papers_df)
    category_stats = create_category_stats(categories_df)
    monthly_category_stats = create_monthly_category_stats(papers_df, categories_df)

    # Keywords
    top_keywords = create_top_keywords(
        papers_df=papers_df,
        top_n=args.top_n_keywords,
        min_df=args.min_df,
        max_features=args.max_features,
    )
    keyword_stats, rising_keywords = create_keyword_stats_and_rising(
        papers_df=papers_df,
        top_n_per_month=args.top_n_per_month,
        min_df=args.min_df,
        max_features=args.max_features,
        min_rising_count=args.min_rising_count,
    )

    # Category synergy
    category_synergy_stats, monthly_category_synergy_stats, category_pair_stats, monthly_category_pair_stats = create_category_synergy_stats(
        papers_df=papers_df,
        categories_df=categories_df,
    )

    # Rule-based paradigms
    rule_paradigm_labels, monthly_rule_paradigm_stats = create_rule_paradigm_labels(papers_df)

    # Model efficiency
    paper_model_efficiency, model_efficiency_stats = create_model_efficiency_stats(papers_df)

    # Curation
    curation_keyword_stats, curation_related_keywords, curation_recent_papers = create_curation_outputs(
        papers_df=papers_df,
        min_df=args.min_df,
        max_features=args.max_features,
    )

    # Co-occurrence graph from top keywords
    keyword_nodes, keyword_edges = create_keyword_cooccurrence_graph(
        papers_df=papers_df,
        top_keywords=top_keywords,
        max_keywords=min(args.top_n_keywords, 100),
    )

    # Probabilistic LDA topics
    if args.skip_topic_model:
        lda_topic_terms = pd.DataFrame(columns=["topic_id", "rank", "term", "weight"])
        paper_topic_probs = pd.DataFrame(columns=["canonical_arxiv_id", "published_month", "dominant_topic", "topic_probability"])
        monthly_topic_stats = pd.DataFrame(columns=["month", "topic_id", "paper_count", "month_total", "ratio", "avg_probability"])
        topic_keyword_labels = pd.DataFrame(columns=["topic_id", "topic_label", "top_terms"])
    else:
        lda_topic_terms, paper_topic_probs, monthly_topic_stats, topic_keyword_labels = create_lda_topic_outputs(
            papers_df=papers_df,
            n_topics=args.n_topics,
            top_terms=args.topic_top_terms,
            min_df=args.min_df,
            max_features=args.max_features,
        )

    kpi_summary = create_kpi_summary(
        papers_df=papers_df,
        categories_df=categories_df,
        monthly_stats=monthly_stats,
        category_stats=category_stats,
        top_keywords=top_keywords,
        rising_keywords=rising_keywords,
    )

    # Save outputs
    save_csv(kpi_summary, output_dir / "kpi_summary.csv")
    save_csv(monthly_stats, output_dir / "monthly_stats.csv")
    save_csv(category_stats, output_dir / "category_stats.csv")
    save_csv(monthly_category_stats, output_dir / "monthly_category_stats.csv")

    save_csv(top_keywords, output_dir / "top_keywords.csv")
    save_csv(keyword_stats, output_dir / "keyword_stats.csv")
    save_csv(rising_keywords, output_dir / "rising_keywords.csv")

    save_csv(category_synergy_stats, output_dir / "category_synergy_stats.csv")
    save_csv(monthly_category_synergy_stats, output_dir / "monthly_category_synergy_stats.csv")
    save_csv(category_pair_stats, output_dir / "category_pair_stats.csv")
    save_csv(monthly_category_pair_stats, output_dir / "monthly_category_pair_stats.csv")

    save_csv(rule_paradigm_labels, output_dir / "rule_paradigm_labels.csv")
    save_csv(monthly_rule_paradigm_stats, output_dir / "monthly_rule_paradigm_stats.csv")

    save_csv(paper_model_efficiency, output_dir / "paper_model_efficiency.csv")
    save_csv(model_efficiency_stats, output_dir / "model_efficiency_stats.csv")

    save_csv(curation_keyword_stats, output_dir / "curation_keyword_stats.csv")
    save_csv(curation_related_keywords, output_dir / "curation_related_keywords.csv")
    save_csv(curation_recent_papers, output_dir / "curation_recent_papers.csv")

    save_csv(keyword_nodes, output_dir / "keyword_cooccurrence_nodes.csv")
    save_csv(keyword_edges, output_dir / "keyword_cooccurrence_edges.csv")

    save_csv(lda_topic_terms, output_dir / "lda_topic_terms.csv")
    save_csv(topic_keyword_labels, output_dir / "topic_keyword_labels.csv")
    save_csv(paper_topic_probs, output_dir / "paper_topic_probs.csv")
    save_csv(monthly_topic_stats, output_dir / "monthly_topic_stats.csv")

    save_run_log(output_dir, papers_df, categories_df, args)

    print("[INFO] Daily rule/stat/probabilistic EDA finished")


def main() -> None:
    args = parse_args()
    run_daily_rule_stat_eda(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user")
        sys.exit(130)
