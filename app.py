import os
import numpy as np
import pandas as pd
import pymysql
import streamlit as st
import plotly.express as px


# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Amazon Competitors PM Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# 2. DB CONFIG
# =========================================================
def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


DB_CONFIG = dict(
    host=get_secret("DB_HOST", "YOUR_DB_HOST"),
    port=int(get_secret("DB_PORT", 3306)),
    user=get_secret("DB_USER", "YOUR_DB_USER"),
    password=get_secret("DB_PASSWORD", "YOUR_DB_PASSWORD"),
    database=get_secret("DB_NAME", "analyticallab"),
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
)


def run_query(sql: str) -> pd.DataFrame:
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [col[0] for col in cur.description] if cur.description else []
        return pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()


# =========================================================
# 3. LOAD DATA
# =========================================================
@st.cache_data(ttl=1800)
def load_data():
    columns_df = run_query("SHOW COLUMNS FROM amazon_competitors")
    available_cols = set(columns_df["Field"].tolist())

    load_candidates = [
        "load_capacity_kg_category",
        "load_capacity_category",
        "max_load_capacity_kg",
        "load_capacity_kg",
        "max_load",
        "weight_capacity",
    ]

    existing_load_cols = [c for c in load_candidates if c in available_cols]

    if existing_load_cols:
        load_expr = "COALESCE(" + ", ".join(existing_load_cols) + ") AS load_category_raw"
    else:
        load_expr = "'' AS load_category_raw"

    sql = f"""
    SELECT
        market,
        type,
        diagonal_category,
        {load_expr},
        onkron_competitor,
        brand,
        asin,
        data_date_begin,
        CAST(price AS DECIMAL(10,2)) AS price,
        CAST(revenue AS DECIMAL(14,2)) AS revenue,
        CAST(sales AS DECIMAL(12,2)) AS sales,
        CAST(rating AS DECIMAL(3,1)) AS rating,
        CAST(reviews AS UNSIGNED) AS reviews,
        CAST(bsr AS DECIMAL(12,2)) AS bsr
    FROM amazon_competitors
    WHERE
        data_date_begin >= '2026-03-01'
        AND data_date_begin < '2026-05-01'
        AND type IS NOT NULL
        AND diagonal_category IS NOT NULL
    """

    df = run_query(sql)

    if df.empty:
        return df

    numeric_cols = ["price", "revenue", "sales", "rating", "reviews", "bsr"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["revenue"] = df["revenue"].fillna(0)
    df["sales"] = df["sales"].fillna(0)
    df["data_date_begin"] = pd.to_datetime(df["data_date_begin"], errors="coerce")
    df["month"] = df["data_date_begin"].dt.to_period("M").astype(str)

    df["load_category"] = df["load_category_raw"].fillna("").astype(str).str.strip()
    df.loc[df["load_category"] == "", "load_category"] = "не указано"

    return df


def norm(s, invert=False):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    mn, mx = s.min(), s.max()

    if mx == mn:
        result = pd.Series([0.5] * len(s), index=s.index)
    else:
        result = (s - mn) / (mx - mn)

    return 1 - result if invert else result


def prepare_score(df):
    keys = ["market", "type", "diagonal_category", "load_category"]

    seg = (
        df.groupby(keys + ["onkron_competitor"], dropna=False)
        .agg(
            revenue=("revenue", "sum"),
            sales=("sales", "sum"),
            players=("asin", "nunique"),
            avg_price=("price", "mean"),
        )
        .reset_index()
    )

    comp = seg[seg["onkron_competitor"] == "competitor"].copy()
    onk = seg[seg["onkron_competitor"] == "onkron"].copy()

    comp_agg = (
        comp.groupby(keys, dropna=False)
        .agg(
            revenue=("revenue", "sum"),
            sales=("sales", "sum"),
            players=("players", "sum"),
            avg_price=("avg_price", "mean"),
        )
        .reset_index()
    )

    onk_agg = (
        onk.groupby(keys, dropna=False)
        .agg(
            onkron_revenue=("revenue", "sum"),
            onkron_sales_units=("sales", "sum"),
            onkron_models=("players", "sum"),
        )
        .reset_index()
    )

    score = comp_agg.merge(onk_agg, on=keys, how="left")

    for col in ["onkron_revenue", "onkron_sales_units", "onkron_models"]:
        score[col] = score[col].fillna(0)

    score["revenue_per_player"] = score["revenue"] / score["players"].replace(0, np.nan)
    score["revenue_per_player"] = score["revenue_per_player"].fillna(0)

    score["onkron_share"] = score["onkron_revenue"] / (score["revenue"] + score["onkron_revenue"] + 1)
    score["onkron_share_pct"] = score["onkron_share"] * 100

    score["opportunity_gap"] = 1 - score["onkron_share"]
    score["opportunity_gap_pct"] = score["opportunity_gap"] * 100

    score["revenue_norm"] = score["revenue"] / score["revenue"].max()
    score["rev_player_norm"] = score["revenue_per_player"] / score["revenue_per_player"].max()

    max_players = score["players"].max()
    score["low_competition_norm"] = 1 - (score["players"] / max_players)

    score["score"] = (
        score["revenue_norm"] * 35
        + score["rev_player_norm"] * 25
        + score["low_competition_norm"] * 20
        + score["opportunity_gap"] * 20
    )

    score["priority"] = np.where(
        score["score"] >= 70,
        "HIGH",
        np.where(score["score"] >= 45, "MEDIUM", "LOW")
    )

    score["status"] = np.where(
        score["onkron_revenue"] > 0,
        "Текущий пайплайн",
        np.where(
            (score["revenue"] >= 50000) & (score["onkron_share_pct"] <= 5),
            "Потенциальный пайплайн",
            "Наблюдать / снизить приоритет"
        )
    )

    score["recommendation"] = np.where(
        score["status"] == "Текущий пайплайн",
        "Масштабировать / защищать позицию",
        np.where(
            score["status"] == "Потенциальный пайплайн",
            "Оценить запуск / закрыть ассортиментный gap",
            "Пока не приоритет"
        )
    )

    return score


# =========================================================
# 4. APP
# =========================================================
st.title("Amazon Competitors — PM Dashboard")
st.caption("Фокус: топ конкурентов, score сегментов, динамика март → апрель, решения ENTER / WATCH / EXIT")

try:
    df = load_data()
except Exception as e:
    st.error("Ошибка подключения к базе или загрузки данных.")
    st.exception(e)
    st.stop()

if df.empty:
    st.error("Нет данных за март–апрель 2026.")
    st.stop()

score_df = prepare_score(df)


# =========================================================
# 5. FILTERS
# =========================================================
markets = sorted(df["market"].dropna().unique())

col1, col2, col3, col4 = st.columns(4)

with col1:
    market = st.selectbox("Рынок", markets)

filtered = df[df["market"] == market].copy()

with col2:
    types = ["Все"] + sorted(filtered["type"].dropna().unique())
    selected_type = st.selectbox("Тип", types)

if selected_type != "Все":
    filtered = filtered[filtered["type"] == selected_type]

with col3:
    diags = ["Все"] + sorted(filtered["diagonal_category"].dropna().unique())
    selected_diag = st.selectbox("Диагональ", diags)

if selected_diag != "Все":
    filtered = filtered[filtered["diagonal_category"] == selected_diag]

with col4:
    loads = ["Все"] + sorted(filtered["load_category"].dropna().unique())
    selected_load = st.selectbox("Нагрузка", loads)

if selected_load != "Все":
    filtered = filtered[filtered["load_category"] == selected_load]


filtered_score = score_df[score_df["market"] == market].copy()

if selected_type != "Все":
    filtered_score = filtered_score[filtered_score["type"] == selected_type]

if selected_diag != "Все":
    filtered_score = filtered_score[filtered_score["diagonal_category"] == selected_diag]

if selected_load != "Все":
    filtered_score = filtered_score[filtered_score["load_category"] == selected_load]


# =========================================================
# 6. KPI
# =========================================================
apr = filtered[filtered["month"] == "2026-04"]
mar = filtered[filtered["month"] == "2026-03"]

apr_revenue = apr["revenue"].sum()
mar_revenue = mar["revenue"].sum()

growth = ((apr_revenue - mar_revenue) / mar_revenue * 100) if mar_revenue else 0

onkron_revenue = apr[apr["onkron_competitor"] == "onkron"]["revenue"].sum()
onkron_share = onkron_revenue / (apr_revenue + 1) * 100

competitor_skus = apr[apr["onkron_competitor"] == "competitor"]["asin"].nunique()

enter_count = int((filtered_score["recommendation"] == "ENTER").sum())
exit_count = int((filtered_score["recommendation"] == "EXIT").sum())

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Оборот апрель", f"€{apr_revenue:,.0f}")
k2.metric("Рост к марту", f"{growth:.1f}%")
k3.metric("Доля Onkron", f"{onkron_share:.1f}%")
k4.metric("SKU конкурентов", f"{competitor_skus:,}")
k5.metric("Сегментов ENTER", enter_count)
k6.metric("Сегментов EXIT", exit_count)


# =========================================================
# 7. CHART 1 — TOP COMPETITORS
# =========================================================
st.subheader("1. Топ конкурентов по обороту")

top_comp = (
    apr[apr["onkron_competitor"] == "competitor"]
    .groupby("brand", dropna=False)
    .agg(
        revenue=("revenue", "sum"),
        skus=("asin", "nunique"),
        avg_price=("price", "mean"),
        avg_rating=("rating", "mean"),
    )
    .reset_index()
    .sort_values("revenue", ascending=False)
    .head(15)
)

if top_comp.empty:
    st.info("Нет данных по конкурентам.")
else:
    fig_top = px.bar(
        top_comp.sort_values("revenue"),
        x="revenue",
        y="brand",
        orientation="h",
        text="revenue",
        hover_data=["skus", "avg_price", "avg_rating"],
        labels={
            "revenue": "Оборот, €",
            "brand": "Бренд",
            "skus": "SKU",
            "avg_price": "Средняя цена",
            "avg_rating": "Средний рейтинг",
        },
    )
    fig_top.update_traces(texttemplate="€%{text:,.0f}", textposition="outside")
    fig_top.update_layout(height=520, margin=dict(l=10, r=80, t=20, b=40))
    st.plotly_chart(fig_top, use_container_width=True)


# =========================================================
# 8. CHART 2 — SCORE
# =========================================================
st.subheader("2. Score сегментов: тип × диагональ × нагрузка")

if filtered_score.empty:
    st.info("Нет сегментов для выбранных фильтров.")
else:
    plot_score = filtered_score.copy()
    plot_score["segment"] = (
        plot_score["type"].astype(str).str[:22]
        + " / "
        + plot_score["diagonal_category"].astype(str)
        + " / "
        + plot_score["load_category"].astype(str)
    )

    fig_score = px.scatter(
        plot_score,
        x="competitor_skus",
        y="score",
        size="competitor_revenue",
        color="recommendation",
        hover_name="segment",
        hover_data={
            "competitor_revenue": ":,.0f",
            "revenue_growth_pct": ":.1f",
            "onkron_share_pct": ":.1f",
            "avg_price": ":.2f",
            "competitor_skus": True,
            "score": ":.1f",
        },
        labels={
            "competitor_skus": "SKU конкурентов",
            "score": "Opportunity Score",
            "competitor_revenue": "Оборот конкурентов",
            "revenue_growth_pct": "Рост оборота, %",
            "onkron_share_pct": "Доля Onkron, %",
            "avg_price": "Средняя цена",
        },
    )

    fig_score.add_hline(y=65, line_dash="dot", annotation_text="ENTER")
    fig_score.add_hline(y=35, line_dash="dot", annotation_text="EXIT")
    fig_score.update_layout(height=520, margin=dict(l=10, r=20, t=20, b=40))

    st.plotly_chart(fig_score, use_container_width=True)


# =========================================================
# 9. CHART 3 — DYNAMICS
# =========================================================
st.subheader("3. Динамика оборота март → апрель")

if filtered_score.empty:
    st.info("Нет данных для динамики.")
else:
    growth_df = filtered_score.copy()
    growth_df["segment"] = (
        growth_df["type"].astype(str).str[:20]
        + " / "
        + growth_df["diagonal_category"].astype(str)
        + " / "
        + growth_df["load_category"].astype(str)
    )

    grown = growth_df.sort_values("revenue_growth_abs", ascending=False).head(10)
    fallen = growth_df.sort_values("revenue_growth_abs", ascending=True).head(10)

    dyn = pd.concat([fallen, grown], ignore_index=True).drop_duplicates("segment")
    dyn = dyn.sort_values("revenue_growth_abs")

    fig_dyn = px.bar(
        dyn,
        x="revenue_growth_abs",
        y="segment",
        orientation="h",
        color="revenue_growth_abs",
        text="revenue_growth_abs",
        hover_data={
            "revenue_2026-03": ":,.0f",
            "revenue_2026-04": ":,.0f",
            "revenue_growth_pct": ":.1f",
        },
        labels={
            "revenue_growth_abs": "Изменение оборота, €",
            "segment": "Сегмент",
            "revenue_2026-03": "Март",
            "revenue_2026-04": "Апрель",
            "revenue_growth_pct": "Рост, %",
        },
    )

    fig_dyn.add_vline(x=0)
    fig_dyn.update_traces(texttemplate="€%{text:,.0f}", textposition="outside")
    fig_dyn.update_layout(height=560, margin=dict(l=10, r=80, t=20, b=40))

    st.plotly_chart(fig_dyn, use_container_width=True)


# =========================================================
# 10. TABLE — DECISIONS
# =========================================================
st.subheader("4. Куда вводить / откуда выводить продукт")

if filtered_score.empty:
    st.info("Нет рекомендаций.")
else:
    table = filtered_score.copy()

    order = {"ENTER": 0, "WATCH": 1, "EXIT": 2}
    table["rec_order"] = table["recommendation"].map(order)

    table = table.sort_values(["rec_order", "score"], ascending=[True, False])

    table = table[
        [
            "type",
            "diagonal_category",
            "revenue",
            "sales",
            "players",
            "avg_price",
            "revenue_per_player",
            "onkron_revenue",
            "onkron_sales_units",
            "onkron_models",
            "onkron_share_pct",
            "opportunity_gap_pct",
            "revenue_norm",
            "rev_player_norm",
            "low_competition_norm",
            "score",
            "priority",
            "status",
            "recommendation",
        ]
    ].copy()

    table.columns = [
        "Type",
        "Diagonal Category",
        "Revenue £",
        "Sales",
        "Players(ASINs)",
        "Avg Price £",
        "Revenue per player £",
        "ONKRON Revenue £",
        "ONKRON Sales units",
        "ONKRON Models",
        "ONKRON Share %",
        "Opportunity Gap %",
        "Revenue Norm",
        "Rev/Player Norm",
        "Low Competition Norm",
        "Score",
        "Приоритет",
        "Статус",
        "Рекомендации",
    ]

    st.dataframe(
        table,
        use_container_width=True,
        height=520
    )


# =========================================================
# 11. FOOTER
# =========================================================
st.caption(
    "Score = оборот конкурентов 30% + рост 25% + средняя цена 15% + низкая конкуренция 15% + низкая доля Onkron 15%. "
    "Период: март–апрель 2026."
)