from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# PaperPulse AI - Integrated Dashboard
# -----------------------------------------------------------------------------
# This Streamlit app serves:
#   1) Daily rule/stat/probabilistic EDA outputs from data/eda/
#   2) Monthly local LLM result EDA outputs from data/eda_llm_derived/
#   3) Raw monthly LLM outputs from data/eda_llm/ when needed
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="PaperPulse AI | Integrated Dashboard",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_RULE_DIR = Path("data") / "eda"
DEFAULT_LLM_DIR = Path("data") / "eda_llm"
DEFAULT_LLM_DERIVED_DIR = Path("data") / "eda_llm_derived"

RULE_FILES = {
    "kpi_summary": "kpi_summary.csv",
    "monthly_stats": "monthly_stats.csv",
    "category_stats": "category_stats.csv",
    "monthly_category_stats": "monthly_category_stats.csv",
    "top_keywords": "top_keywords.csv",
    "keyword_stats": "keyword_stats.csv",
    "rising_keywords": "rising_keywords.csv",
    "category_synergy_stats": "category_synergy_stats.csv",
    "monthly_category_synergy_stats": "monthly_category_synergy_stats.csv",
    "category_pair_stats": "category_pair_stats.csv",
    "monthly_category_pair_stats": "monthly_category_pair_stats.csv",
    "rule_paradigm_labels": "rule_paradigm_labels.csv",
    "monthly_rule_paradigm_stats": "monthly_rule_paradigm_stats.csv",
    "paper_model_efficiency": "paper_model_efficiency.csv",
    "model_efficiency_stats": "model_efficiency_stats.csv",
    "curation_keyword_stats": "curation_keyword_stats.csv",
    "curation_related_keywords": "curation_related_keywords.csv",
    "curation_recent_papers": "curation_recent_papers.csv",
    "keyword_cooccurrence_nodes": "keyword_cooccurrence_nodes.csv",
    "keyword_cooccurrence_edges": "keyword_cooccurrence_edges.csv",
    "lda_topic_terms": "lda_topic_terms.csv",
    "topic_keyword_labels": "topic_keyword_labels.csv",
    "paper_topic_probs": "paper_topic_probs.csv",
    "monthly_topic_stats": "monthly_topic_stats.csv",
    "daily_eda_run_log": "daily_eda_run_log.csv",
}

LLM_RAW_FILES = {
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

LLM_DERIVED_FILES = {
    "llm_master_table": "llm_master_table.csv",
    "llm_paper_concepts_enriched": "llm_paper_concepts_enriched.csv",
    "llm_kpi_summary": "llm_kpi_summary.csv",
    "llm_paradigm_stats": "llm_paradigm_stats.csv",
    "llm_monthly_paradigm_stats": "llm_monthly_paradigm_stats.csv",
    "llm_contribution_stats": "llm_contribution_stats.csv",
    "llm_category_paradigm_matrix": "llm_category_paradigm_matrix.csv",
    "llm_confidence_stats": "llm_confidence_stats.csv",
    "llm_confidence_bins": "llm_confidence_bins.csv",
    "llm_low_confidence_papers": "llm_low_confidence_papers.csv",
    "llm_keyword_stats": "llm_keyword_stats.csv",
    "llm_concept_stats": "llm_concept_stats.csv",
    "llm_monthly_concept_stats": "llm_monthly_concept_stats.csv",
    "llm_concept_type_stats": "llm_concept_type_stats.csv",
    "llm_concept_paradigm_matrix": "llm_concept_paradigm_matrix.csv",
    "llm_concept_category_matrix": "llm_concept_category_matrix.csv",
    "llm_concept_edges_ranked": "llm_concept_edges_ranked.csv",
    "llm_representative_papers": "llm_representative_papers.csv",
    "llm_insight_text_stats": "llm_insight_text_stats.csv",
    "llm_result_eda_run_log": "llm_result_eda_run_log.csv",
}


# -----------------------------------------------------------------------------
# Style and loading
# -----------------------------------------------------------------------------


def add_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.3rem; padding-bottom: 2rem; }
        .small-caption { color: #6b7280; font-size: 0.88rem; }
        .section-note {
            border-left: 4px solid #d1d5db;
            padding: 0.55rem 0.8rem;
            background: rgba(249,250,251,0.9);
            color: #374151;
            font-size: 0.94rem;
            margin-bottom: 1rem;
        }
        .paper-card {
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 14px;
            padding: 15px 18px;
            margin-top: 10px;
            background: rgba(250, 250, 250, 0.72);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_csv_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"CSV 로드 실패: {path} / {exc}")
        return pd.DataFrame()


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()

    text_cols = [
        "canonical_arxiv_id",
        "month",
        "published_month",
        "primary_category",
        "category",
        "primary_paradigm",
        "rule_primary_paradigm",
        "contribution_type",
        "concept_name",
        "concept_type",
        "keyword",
        "title",
        "problem",
        "method",
        "result",
        "abs_url",
        "pdf_url",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "", "NaT": ""}).str.strip()

    numeric_cols = [
        "paper_count",
        "keyword_count",
        "previous_month_count",
        "growth_rate",
        "trend_score",
        "category_ratio",
        "ratio",
        "ratio_by_category",
        "month_total",
        "avg_confidence",
        "confidence",
        "mean",
        "median",
        "min",
        "max",
        "edge_weight",
        "rank",
        "rank_in_month",
        "topic_probability",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner="EDA 산출물을 불러오는 중입니다...")
def load_all_data(rule_dir_str: str, llm_dir_str: str, llm_derived_dir_str: str) -> dict[str, Any]:
    rule_dir = Path(rule_dir_str)
    llm_dir = Path(llm_dir_str)
    llm_derived_dir = Path(llm_derived_dir_str)

    rule = {}
    llm_raw = {}
    llm = {}
    status_rows = []

    for group_name, base_dir, file_map, target_dict in [
        ("Daily Rule/Stat", rule_dir, RULE_FILES, rule),
        ("Monthly LLM Raw", llm_dir, LLM_RAW_FILES, llm_raw),
        ("Monthly LLM Derived", llm_derived_dir, LLM_DERIVED_FILES, llm),
    ]:
        for key, filename in file_map.items():
            path = base_dir / filename
            exists = path.exists()
            df = normalize_df(read_csv_cached(str(path))) if exists else pd.DataFrame()
            target_dict[key] = df
            status_rows.append(
                {
                    "group": group_name,
                    "key": key,
                    "filename": filename,
                    "exists": exists,
                    "rows": len(df),
                    "path": str(path),
                }
            )

    return {
        "rule_dir": rule_dir,
        "llm_dir": llm_dir,
        "llm_derived_dir": llm_derived_dir,
        "rule": rule,
        "llm_raw": llm_raw,
        "llm": llm,
        "status": pd.DataFrame(status_rows),
    }


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def filter_by_month(df: pd.DataFrame, selected_month: str) -> pd.DataFrame:
    if df.empty or selected_month == "전체":
        return df.copy()
    if "published_month" in df.columns:
        return df[df["published_month"].astype(str) == selected_month].copy()
    if "month" in df.columns:
        return df[df["month"].astype(str) == selected_month].copy()
    return df.copy()


def get_available_months(data: dict[str, Any]) -> list[str]:
    months: set[str] = set()
    for group in ["rule", "llm", "llm_raw"]:
        for df in data[group].values():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            for col in ["published_month", "month"]:
                if col in df.columns:
                    months.update(
                        value for value in df[col].dropna().astype(str).tolist()
                        if re.fullmatch(r"\d{4}-\d{2}", value)
                    )
    return sorted(months)


def kpi_to_dict(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or not {"metric_name", "metric_value"}.issubset(df.columns):
        return {}
    return dict(zip(df["metric_name"].astype(str), df["metric_value"]))


def metric_value(kpis: dict[str, Any], key: str, default: Any = "-") -> Any:
    value = kpis.get(key, default)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def safe_top(df: pd.DataFrame, col: str) -> tuple[str, int]:
    if df.empty or col not in df.columns:
        return "-", 0
    counts = df[col].dropna().astype(str).value_counts()
    if counts.empty:
        return "-", 0
    return str(counts.index[0]), int(counts.iloc[0])


def safe_count(df: pd.DataFrame, id_col: str = "canonical_arxiv_id") -> int:
    if df.empty:
        return 0
    if id_col in df.columns:
        return int(df[id_col].dropna().nunique())
    return len(df)


def require_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return bool(not df.empty and set(columns).issubset(df.columns))


def show_missing(message: str, command: str | None = None) -> None:
    st.info(message)
    if command:
        st.code(command, language="bash")


def render_dataframe(df: pd.DataFrame, height: int = 360, use_container_width: bool = True) -> None:
    st.dataframe(df, use_container_width=use_container_width, height=height)


# -----------------------------------------------------------------------------
# Chart helpers
# -----------------------------------------------------------------------------


def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    orientation: str = "v",
    height: int = 420,
) -> None:
    if not require_columns(df, [x, y]):
        show_missing(f"차트에 필요한 컬럼이 없습니다: {x}, {y}")
        return
    fig = px.bar(df, x=x, y=y, color=color, orientation=orientation, title=title)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plot_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    markers: bool = True,
    height: int = 420,
) -> None:
    if not require_columns(df, [x, y]):
        show_missing(f"차트에 필요한 컬럼이 없습니다: {x}, {y}")
        return
    fig = px.line(df, x=x, y=y, color=color, markers=markers, title=title)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plot_heatmap(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    title: str,
    height: int = 520,
) -> None:
    if not require_columns(df, [index, columns, values]):
        show_missing(f"히트맵에 필요한 컬럼이 없습니다: {index}, {columns}, {values}")
        return

    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc="sum", fill_value=0)
    if pivot.empty:
        show_missing("히트맵에 표시할 데이터가 없습니다.")
        return

    fig = px.imshow(pivot, text_auto=True, aspect="auto", title=title)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Network graph
# -----------------------------------------------------------------------------


def build_concept_network(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    top_n_nodes: int,
    min_edge_weight: int,
) -> go.Figure | None:
    if nodes_df.empty or edges_df.empty:
        return None
    if not {"concept_name", "paper_count"}.issubset(nodes_df.columns):
        return None
    if not {"source", "target", "edge_weight"}.issubset(edges_df.columns):
        return None

    node_subset = nodes_df.sort_values("paper_count", ascending=False).head(top_n_nodes).copy()
    allowed = set(node_subset["concept_name"].astype(str))

    edge_subset = edges_df[
        edges_df["source"].astype(str).isin(allowed)
        & edges_df["target"].astype(str).isin(allowed)
        & (pd.to_numeric(edges_df["edge_weight"], errors="coerce").fillna(0) >= min_edge_weight)
    ].copy()

    if edge_subset.empty:
        return None

    graph = nx.Graph()

    for _, row in node_subset.iterrows():
        concept = str(row["concept_name"])
        graph.add_node(
            concept,
            paper_count=float(row.get("paper_count", 1) or 1),
            concept_type=str(row.get("concept_type", "other")),
        )

    for _, row in edge_subset.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        if source in graph and target in graph:
            graph.add_edge(source, target, weight=float(row.get("edge_weight", 1) or 1))

    if graph.number_of_edges() == 0:
        return None

    pos = nx.spring_layout(graph, k=0.75, iterations=80, seed=42, weight="weight")

    edge_x = []
    edge_y = []
    edge_hover = []

    for source, target, attrs in graph.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_hover.append(f"{source} - {target}: {attrs.get('weight', 1):.0f}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.7),
        hoverinfo="none",
        mode="lines",
        name="co-occurrence",
    )

    node_x = []
    node_y = []
    node_size = []
    node_text = []
    node_color = []

    type_to_id = {}

    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        count = float(attrs.get("paper_count", 1) or 1)
        concept_type = str(attrs.get("concept_type", "other"))
        if concept_type not in type_to_id:
            type_to_id[concept_type] = len(type_to_id) + 1
        node_x.append(x)
        node_y.append(y)
        node_size.append(12 + math.sqrt(count) * 8)
        node_color.append(type_to_id[concept_type])
        node_text.append(f"{node}<br>type: {concept_type}<br>papers: {count:.0f}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[text.split("<br>")[0] for text in node_text],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            showscale=False,
            line=dict(width=1),
        ),
        name="concepts",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="LLM Concept Co-occurrence Network",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=10, r=10, t=55, b=10),
        height=640,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


# -----------------------------------------------------------------------------
# Tab renderers
# -----------------------------------------------------------------------------


def render_overview(data: dict[str, Any], selected_month: str) -> None:
    st.subheader("Integrated Overview")
    st.markdown(
        "<div class='section-note'>일간 통계·규칙 기반 EDA와 월간 로컬 LLM 기반 EDA를 함께 요약한다.</div>",
        unsafe_allow_html=True,
    )

    rule = data["rule"]
    llm = data["llm"]
    rule_kpi = kpi_to_dict(rule.get("kpi_summary", pd.DataFrame()))
    llm_kpi = kpi_to_dict(llm.get("llm_kpi_summary", pd.DataFrame()))

    llm_master = filter_by_month(llm.get("llm_master_table", pd.DataFrame()), selected_month)
    monthly_stats = filter_by_month(rule.get("monthly_stats", pd.DataFrame()), selected_month)
    monthly_full = rule.get("monthly_stats", pd.DataFrame())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rule/Stat 전체 논문", metric_value(rule_kpi, "total_papers", safe_count(monthly_full, "canonical_arxiv_id")))
    c2.metric("LLM 분석 논문", metric_value(llm_kpi, "llm_total_papers", safe_count(llm_master)))
    c3.metric("Top LLM Paradigm", metric_value(llm_kpi, "llm_top_paradigm", "-"))
    c4.metric("Top LLM Concept", metric_value(llm_kpi, "llm_top_concept", "-"))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("LLM 평균 confidence", metric_value(llm_kpi, "llm_avg_confidence", "-"))
    c6.metric("LLM Concept 수", metric_value(llm_kpi, "llm_total_concepts", "-"))
    c7.metric("Rule Top Keyword", metric_value(rule_kpi, "top_keyword", "-"))
    c8.metric("선택 월", selected_month)

    left, right = st.columns([1.15, 1])

    with left:
        if not monthly_full.empty and require_columns(monthly_full, ["month", "paper_count"]):
            plot_line(monthly_full.sort_values("month"), "month", "paper_count", "Rule/Stat 월별 논문 수")
        else:
            show_missing(
                "월별 논문 수 통계가 없습니다.",
                "python3 daily_rule_stat_eda.py",
            )

    with right:
        paradigm_stats = filter_by_month(llm.get("llm_paradigm_stats", pd.DataFrame()), selected_month)
        if not paradigm_stats.empty:
            plot_bar(
                paradigm_stats.sort_values("paper_count", ascending=True),
                x="paper_count",
                y="primary_paradigm",
                title="LLM 연구 패러다임 분포",
                orientation="h",
                height=420,
            )
        else:
            show_missing(
                "LLM 파생 EDA가 없습니다. 먼저 LLM 결과 EDA를 실행하세요.",
                "python3 llm_result_eda.py --target-month 2026-07",
            )

    left, right = st.columns(2)
    with left:
        rising = rule.get("rising_keywords", pd.DataFrame())
        rising = filter_by_month(rising, selected_month)
        score_col = "trend_score" if "trend_score" in rising.columns else "growth_rate"
        if require_columns(rising, ["keyword", score_col]):
            plot_bar(
                rising.sort_values(score_col, ascending=False).head(20).sort_values(score_col),
                x=score_col,
                y="keyword",
                title="Rule/Stat 급상승 키워드 Top 20",
                orientation="h",
            )
        else:
            show_missing("급상승 키워드 파일이 없거나 컬럼이 부족합니다.")

    with right:
        llm_concepts = filter_by_month(llm.get("llm_concept_stats", pd.DataFrame()), selected_month)
        if require_columns(llm_concepts, ["concept_name", "paper_count"]):
            plot_bar(
                llm_concepts.sort_values("paper_count", ascending=False).head(20).sort_values("paper_count"),
                x="paper_count",
                y="concept_name",
                title="LLM 핵심 Concept Top 20",
                orientation="h",
            )
        else:
            show_missing("LLM concept 통계가 없습니다.")


def render_rule_stat_tab(data: dict[str, Any], selected_month: str) -> None:
    st.subheader("Daily Rule/Stat EDA")
    st.markdown(
        "<div class='section-note'>LLM을 사용하지 않는 통계·규칙·확률 기반 분석 결과다. 매일 실행하는 배치 결과를 서빙한다.</div>",
        unsafe_allow_html=True,
    )

    rule = data["rule"]

    col1, col2 = st.columns(2)
    with col1:
        category_stats = rule.get("category_stats", pd.DataFrame())
        if require_columns(category_stats, ["category", "paper_count"]):
            plot_bar(
                category_stats.sort_values("paper_count", ascending=True),
                x="paper_count",
                y="category",
                title="카테고리별 논문 수",
                orientation="h",
            )
        else:
            show_missing("category_stats.csv가 없거나 컬럼이 부족합니다.")

    with col2:
        monthly_rule = rule.get("monthly_rule_paradigm_stats", pd.DataFrame())
        monthly_rule = filter_by_month(monthly_rule, selected_month)
        if require_columns(monthly_rule, ["month", "rule_primary_paradigm", "paper_count"]):
            fig = px.bar(
                monthly_rule,
                x="month",
                y="paper_count",
                color="rule_primary_paradigm",
                title="규칙 기반 연구 패러다임 월별 분포",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_missing("monthly_rule_paradigm_stats.csv가 없거나 컬럼이 부족합니다.")

    st.markdown("### Keyword and Topic Analysis")
    col3, col4 = st.columns(2)

    with col3:
        top_keywords = rule.get("top_keywords", pd.DataFrame())
        if require_columns(top_keywords, ["keyword", "keyword_count"]):
            plot_bar(
                top_keywords.sort_values("keyword_count", ascending=False).head(25).sort_values("keyword_count"),
                x="keyword_count",
                y="keyword",
                title="전체 상위 키워드",
                orientation="h",
            )
        else:
            show_missing("top_keywords.csv가 없거나 컬럼이 부족합니다.")

    with col4:
        topic_labels = rule.get("topic_keyword_labels", pd.DataFrame())
        if not topic_labels.empty:
            st.markdown("#### LDA 토픽 라벨")
            render_dataframe(topic_labels.head(30), height=420)
        else:
            show_missing("LDA 토픽 결과가 없습니다. 필요한 경우 daily_rule_stat_eda.py를 --skip-topic-model 없이 실행하세요.")

    st.markdown("### Category Synergy")
    synergy = rule.get("category_synergy_stats", pd.DataFrame())
    pair_stats = rule.get("category_pair_stats", pd.DataFrame())

    col5, col6 = st.columns(2)
    with col5:
        if require_columns(synergy, ["category_combination", "paper_count"]):
            plot_bar(
                synergy.sort_values("paper_count", ascending=True),
                x="paper_count",
                y="category_combination",
                title="카테고리 조합별 논문 수",
                orientation="h",
            )
        else:
            show_missing("category_synergy_stats.csv가 없거나 컬럼이 부족합니다.")

    with col6:
        if require_columns(pair_stats, ["source_category", "target_category", "paper_count"]):
            plot_heatmap(
                pair_stats,
                index="source_category",
                columns="target_category",
                values="paper_count",
                title="카테고리 Pair Co-occurrence Heatmap",
            )
        else:
            show_missing("category_pair_stats.csv가 없거나 컬럼이 부족합니다.")

    st.markdown("### Model Efficiency Signals")
    model_eff = rule.get("model_efficiency_stats", pd.DataFrame())
    if not model_eff.empty:
        render_dataframe(model_eff.head(80), height=420)
    else:
        show_missing("model_efficiency_stats.csv가 없습니다.")


def render_llm_tab(data: dict[str, Any], selected_month: str) -> None:
    st.subheader("Monthly Local LLM EDA")
    st.markdown(
        "<div class='section-note'>Qwen 로컬 LLM이 월 1회 생성한 논문 의미 분석 결과를 집계한 화면이다.</div>",
        unsafe_allow_html=True,
    )

    llm = data["llm"]

    col1, col2 = st.columns(2)
    with col1:
        paradigm_stats = filter_by_month(llm.get("llm_paradigm_stats", pd.DataFrame()), selected_month)
        if require_columns(paradigm_stats, ["primary_paradigm", "paper_count"]):
            plot_bar(
                paradigm_stats.sort_values("paper_count", ascending=True),
                x="paper_count",
                y="primary_paradigm",
                title="LLM 연구 패러다임 분포",
                orientation="h",
            )
        else:
            show_missing("llm_paradigm_stats.csv가 없습니다.", "python3 llm_result_eda.py --target-month 2026-07")

    with col2:
        contribution = filter_by_month(llm.get("llm_contribution_stats", pd.DataFrame()), selected_month)
        if require_columns(contribution, ["contribution_type", "paper_count"]):
            fig = px.pie(
                contribution,
                names="contribution_type",
                values="paper_count",
                title="LLM Contribution Type 분포",
                hole=0.35,
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_missing("llm_contribution_stats.csv가 없습니다.")

    col3, col4 = st.columns(2)
    with col3:
        monthly_paradigm = llm.get("llm_monthly_paradigm_stats", pd.DataFrame())
        monthly_paradigm = filter_by_month(monthly_paradigm, selected_month)
        if require_columns(monthly_paradigm, ["month", "primary_paradigm", "paper_count"]):
            fig = px.bar(
                monthly_paradigm,
                x="month",
                y="paper_count",
                color="primary_paradigm",
                title="LLM 패러다임 월별 분포",
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_missing("llm_monthly_paradigm_stats.csv가 없습니다.")

    with col4:
        confidence_bins = filter_by_month(llm.get("llm_confidence_bins", pd.DataFrame()), selected_month)
        if require_columns(confidence_bins, ["confidence_bin", "paper_count"]):
            plot_bar(confidence_bins, "confidence_bin", "paper_count", "LLM Confidence 분포")
        else:
            show_missing("llm_confidence_bins.csv가 없습니다.")

    st.markdown("### Category × Paradigm Matrix")
    matrix = filter_by_month(llm.get("llm_category_paradigm_matrix", pd.DataFrame()), selected_month)
    if require_columns(matrix, ["primary_category", "primary_paradigm", "paper_count"]):
        plot_heatmap(
            matrix,
            index="primary_category",
            columns="primary_paradigm",
            values="paper_count",
            title="LLM Category × Paradigm Heatmap",
        )
    else:
        show_missing("llm_category_paradigm_matrix.csv가 없습니다.")

    st.markdown("### Concepts and LLM Keywords")
    col5, col6 = st.columns(2)
    with col5:
        concept_stats = filter_by_month(llm.get("llm_concept_stats", pd.DataFrame()), selected_month)
        if require_columns(concept_stats, ["concept_name", "paper_count"]):
            plot_bar(
                concept_stats.sort_values("paper_count", ascending=False).head(30).sort_values("paper_count"),
                x="paper_count",
                y="concept_name",
                title="LLM Concept Top 30",
                orientation="h",
                height=520,
            )
        else:
            show_missing("llm_concept_stats.csv가 없습니다.")

    with col6:
        keyword_stats = filter_by_month(llm.get("llm_keyword_stats", pd.DataFrame()), selected_month)
        if require_columns(keyword_stats, ["keyword", "paper_count"]):
            plot_bar(
                keyword_stats.sort_values("paper_count", ascending=False).head(30).sort_values("paper_count"),
                x="paper_count",
                y="keyword",
                title="LLM Extracted Keywords Top 30",
                orientation="h",
                height=520,
            )
        else:
            show_missing("llm_keyword_stats.csv가 없습니다.")


def render_network_tab(data: dict[str, Any], selected_month: str, top_n_nodes: int, min_edge_weight: int) -> None:
    st.subheader("Concept and Keyword Networks")
    st.markdown(
        "<div class='section-note'>LLM concept graph와 규칙 기반 키워드 co-occurrence graph를 비교한다.</div>",
        unsafe_allow_html=True,
    )

    llm = data["llm"]
    rule = data["rule"]

    concept_nodes = filter_by_month(llm.get("llm_concept_stats", pd.DataFrame()), selected_month)
    concept_edges = filter_by_month(llm.get("llm_concept_edges_ranked", pd.DataFrame()), selected_month)

    fig = build_concept_network(
        nodes_df=concept_nodes,
        edges_df=concept_edges,
        top_n_nodes=top_n_nodes,
        min_edge_weight=min_edge_weight,
    )

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing("LLM concept graph를 그릴 충분한 node/edge 데이터가 없습니다.")

    st.markdown("### Rule-based Keyword Co-occurrence")
    rule_nodes = filter_by_month(rule.get("keyword_cooccurrence_nodes", pd.DataFrame()), selected_month)
    rule_edges = filter_by_month(rule.get("keyword_cooccurrence_edges", pd.DataFrame()), selected_month)

    if not rule_nodes.empty:
        st.markdown("#### Keyword Nodes")
        render_dataframe(rule_nodes.head(80), height=300)
    else:
        show_missing("keyword_cooccurrence_nodes.csv가 없습니다.")

    if not rule_edges.empty:
        st.markdown("#### Keyword Edges")
        render_dataframe(rule_edges.head(120), height=340)
    else:
        show_missing("keyword_cooccurrence_edges.csv가 없습니다.")


def render_explorer_tab(data: dict[str, Any], selected_month: str) -> None:
    st.subheader("Paper Explorer")
    st.markdown(
        "<div class='section-note'>LLM이 구조화한 문제·방법·결과 요약을 기준으로 논문을 탐색한다.</div>",
        unsafe_allow_html=True,
    )

    master = filter_by_month(data["llm"].get("llm_master_table", pd.DataFrame()), selected_month)

    if master.empty:
        show_missing("llm_master_table.csv가 없습니다.", "python3 llm_result_eda.py --target-month 2026-07")
        return

    paradigms = sorted([p for p in master.get("primary_paradigm", pd.Series(dtype=str)).dropna().astype(str).unique() if p])
    categories = sorted([c for c in master.get("primary_category", pd.Series(dtype=str)).dropna().astype(str).unique() if c])
    contribution_types = sorted([c for c in master.get("contribution_type", pd.Series(dtype=str)).dropna().astype(str).unique() if c])

    f1, f2, f3 = st.columns(3)
    with f1:
        selected_paradigms = st.multiselect("Paradigm", paradigms, default=paradigms)
    with f2:
        selected_categories = st.multiselect("Primary category", categories, default=categories)
    with f3:
        selected_contributions = st.multiselect("Contribution type", contribution_types, default=contribution_types)

    search_query = st.text_input("검색어", value="", placeholder="title, problem, method, result, concept keyword")

    filtered = master.copy()

    if selected_paradigms and "primary_paradigm" in filtered.columns:
        filtered = filtered[filtered["primary_paradigm"].isin(selected_paradigms)]
    if selected_categories and "primary_category" in filtered.columns:
        filtered = filtered[filtered["primary_category"].isin(selected_categories)]
    if selected_contributions and "contribution_type" in filtered.columns:
        filtered = filtered[filtered["contribution_type"].isin(selected_contributions)]

    if search_query.strip():
        pattern = re.escape(search_query.strip().lower())
        searchable_cols = [col for col in ["title", "summary", "problem", "method", "result", "keywords", "concepts_json"] if col in filtered.columns]
        if searchable_cols:
            mask = pd.Series(False, index=filtered.index)
            for col in searchable_cols:
                mask = mask | filtered[col].fillna("").astype(str).str.lower().str.contains(pattern, regex=True)
            filtered = filtered[mask]

    st.caption(f"표시 논문 수: {len(filtered)}")

    visible_cols = [
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
    visible_cols = [col for col in visible_cols if col in filtered.columns]

    render_dataframe(filtered[visible_cols].head(300), height=430)

    if filtered.empty:
        return

    st.markdown("### 상세 보기")
    id_options = filtered["canonical_arxiv_id"].astype(str).tolist()
    selected_id = st.selectbox("논문 선택", id_options)
    row = filtered[filtered["canonical_arxiv_id"].astype(str) == selected_id].iloc[0]

    st.markdown("<div class='paper-card'>", unsafe_allow_html=True)
    st.markdown(f"#### {row.get('title', '')}")
    st.markdown(
        f"**arXiv ID:** `{row.get('canonical_arxiv_id', '')}`  \n"
        f"**Month:** {row.get('published_month', '')}  \n"
        f"**Category:** {row.get('primary_category', '')}  \n"
        f"**Paradigm:** {row.get('primary_paradigm', '')}  \n"
        f"**Contribution:** {row.get('contribution_type', '')}  \n"
        f"**Confidence:** {row.get('confidence', '')}"
    )

    st.markdown("**문제**")
    st.write(row.get("problem", ""))
    st.markdown("**방법**")
    st.write(row.get("method", ""))
    st.markdown("**결과**")
    st.write(row.get("result", ""))

    if str(row.get("abs_url", "")).startswith("http"):
        st.link_button("arXiv abstract", str(row.get("abs_url")))
    if str(row.get("pdf_url", "")).startswith("http"):
        st.link_button("PDF", str(row.get("pdf_url")))

    with st.expander("원문 초록 보기"):
        st.write(row.get("summary", ""))

    st.markdown("</div>", unsafe_allow_html=True)


def render_comparison_tab(data: dict[str, Any], selected_month: str) -> None:
    st.subheader("Comparison and Diagnostics")
    st.markdown(
        "<div class='section-note'>규칙 기반 분류와 LLM 분류의 차이를 비교하고, 산출물 상태를 점검한다.</div>",
        unsafe_allow_html=True,
    )

    rule_labels = filter_by_month(data["rule"].get("rule_paradigm_labels", pd.DataFrame()), selected_month)
    llm_master = filter_by_month(data["llm"].get("llm_master_table", pd.DataFrame()), selected_month)

    if require_columns(rule_labels, ["canonical_arxiv_id", "rule_primary_paradigm"]) and require_columns(llm_master, ["canonical_arxiv_id", "primary_paradigm"]):
        compare = rule_labels[["canonical_arxiv_id", "rule_primary_paradigm"]].merge(
            llm_master[["canonical_arxiv_id", "primary_paradigm", "confidence"]],
            on="canonical_arxiv_id",
            how="inner",
        )
        compare["is_same_label"] = compare["rule_primary_paradigm"] == compare["primary_paradigm"]

        c1, c2, c3 = st.columns(3)
        c1.metric("비교 가능 논문", len(compare))
        c2.metric("일치 논문", int(compare["is_same_label"].sum()))
        agreement = round(float(compare["is_same_label"].mean() * 100), 2) if len(compare) else 0.0
        c3.metric("라벨 일치율", f"{agreement}%")

        confusion = (
            compare.groupby(["rule_primary_paradigm", "primary_paradigm"])
            .agg(paper_count=("canonical_arxiv_id", "nunique"))
            .reset_index()
        )
        plot_heatmap(
            confusion,
            index="rule_primary_paradigm",
            columns="primary_paradigm",
            values="paper_count",
            title="Rule Paradigm × LLM Paradigm",
            height=620,
        )

        with st.expander("비교 원본 테이블"):
            render_dataframe(compare, height=360)
    else:
        show_missing("rule_paradigm_labels.csv 또는 llm_master_table.csv가 부족하여 비교할 수 없습니다.")

    st.markdown("### Low Confidence Papers")
    low_conf = filter_by_month(data["llm"].get("llm_low_confidence_papers", pd.DataFrame()), selected_month)
    if not low_conf.empty:
        render_dataframe(low_conf.head(120), height=360)
    else:
        st.caption("낮은 confidence 논문이 없거나 low confidence 파일이 없습니다.")

    st.markdown("### File Status")
    status = data["status"].copy()
    render_dataframe(status, height=430)

    st.markdown("### Run Logs")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Daily EDA")
        daily_log = data["rule"].get("daily_eda_run_log", pd.DataFrame())
        if not daily_log.empty:
            render_dataframe(daily_log.tail(10), height=240)
        else:
            st.caption("daily_eda_run_log.csv 없음")
    with col2:
        st.markdown("#### LLM Raw")
        llm_log = data["llm_raw"].get("llm_run_log", pd.DataFrame())
        if not llm_log.empty:
            render_dataframe(llm_log.tail(10), height=240)
        else:
            st.caption("llm_run_log.csv 없음")
    with col3:
        st.markdown("#### LLM Result EDA")
        derived_log = data["llm"].get("llm_result_eda_run_log", pd.DataFrame())
        if not derived_log.empty:
            render_dataframe(derived_log.tail(10), height=240)
        else:
            st.caption("llm_result_eda_run_log.csv 없음")


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------


def main() -> None:
    add_style()

    st.title("PaperPulse AI Integrated Dashboard")
    st.caption("Daily rule/stat EDA + Monthly local LLM EDA")

    with st.sidebar:
        st.header("Data Sources")
        rule_dir = st.text_input("Rule/Stat EDA directory", value=str(DEFAULT_RULE_DIR))
        llm_dir = st.text_input("LLM raw directory", value=str(DEFAULT_LLM_DIR))
        llm_derived_dir = st.text_input("LLM derived EDA directory", value=str(DEFAULT_LLM_DERIVED_DIR))

        if st.button("데이터 새로고침"):
            st.cache_data.clear()
            st.rerun()

    data = load_all_data(rule_dir, llm_dir, llm_derived_dir)

    months = get_available_months(data)
    month_options = ["전체"] + months

    with st.sidebar:
        st.header("Filters")
        default_index = len(month_options) - 1 if len(month_options) > 1 else 0
        selected_month = st.selectbox("Month", month_options, index=default_index)
        top_n_nodes = st.slider("Concept graph Top N nodes", min_value=10, max_value=100, value=45, step=5)
        min_edge_weight = st.slider("Concept graph min edge weight", min_value=1, max_value=10, value=1, step=1)

        st.header("Batch Commands")
        st.code("python3 daily_rule_stat_eda.py", language="bash")
        st.code("python3 monthly_llm_eda.py --target-month 2026-07 --max-papers 300", language="bash")
        st.code("python3 llm_result_eda.py --target-month 2026-07", language="bash")

    if not Path(llm_derived_dir).exists():
        st.warning("LLM 파생 EDA 디렉터리가 없습니다. llm_result_eda.py를 먼저 실행하세요.")
        st.code("python3 llm_result_eda.py --target-month 2026-07", language="bash")

    tabs = st.tabs(
        [
            "Overview",
            "Rule/Stat EDA",
            "LLM EDA",
            "Networks",
            "Paper Explorer",
            "Comparison & Diagnostics",
        ]
    )

    with tabs[0]:
        render_overview(data, selected_month)
    with tabs[1]:
        render_rule_stat_tab(data, selected_month)
    with tabs[2]:
        render_llm_tab(data, selected_month)
    with tabs[3]:
        render_network_tab(data, selected_month, top_n_nodes, min_edge_weight)
    with tabs[4]:
        render_explorer_tab(data, selected_month)
    with tabs[5]:
        render_comparison_tab(data, selected_month)


if __name__ == "__main__":
    main()
