import os
import numpy as np
import pandas as pd
import pymysql
import streamlit as st
import plotly.express as px


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Куда вводить продукт",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# SETTINGS
# =========================================================
PERIOD_START = "2026-04-01"
PERIOD_END = "2026-05-01"
PERIOD_LABEL = "апрель 2026"

WEIGHT_REVENUE = 0.35
WEIGHT_REV_PLAYER = 0.25
WEIGHT_LOW_COMPETITION = 0.20
WEIGHT_OPPORTUNITY_GAP = 0.20

THRESHOLD_HIGH = 70
THRESHOLD_MEDIUM = 45

PIPELINE_MIN_REVENUE = 50000
PIPELINE_MAX_ONKRON_SHARE = 5.0


# =========================================================
# DB CONFIG
# =========================================================
def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


DB_CONFIG = dict(
    host=get_secret("DB_HOST"),
    port=int(get_secret("DB_PORT", 3306)),
    user=get_secret("DB_USER"),
    password=get_secret("DB_PASSWORD"),
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
            cols = [c[0] for c in cur.description] if cur.description else []
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


# =========================================================
# FORMAT HELPERS
# =========================================================
def currency_symbol(market):
    market = str(market).lower()

    # UK
    if (
        "uk" in market
        or "co.uk" in market
        or "united kingdom" in market
        or "britain" in market
        or "amazon uk" in market
    ):
        return "£"

    # USA
    if (
        "usa" in market
        or market == "us"
        or "amazon.com" in market
        or ".com" in market
    ):
        return "$"

    # Russia
    if (
        "ru" in market
        or "russia" in market
        or "рос" in market
        or "ozon" in market
        or "wildberries" in market
        or "wb" in market
        or "яндекс" in market
        or "yandex" in market
    ):
        return "RUB"

    # Germany / France / Italy / Spain / Europe
    return "€"


def money_fmt(value, symbol):
    value = 0 if pd.isna(value) else float(value)

    if symbol == "RUB":
        return f"{value:,.2f} RUB"

    if symbol == "£":
        return f"£ {value:,.2f}"

    if symbol == "$":
        return f"$ {value:,.2f}"

    return f"€ {value:,.2f}"


def int_fmt(value):
    value = 0 if pd.isna(value) else float(value)
    return f"{value:,.0f}"


def pct_fmt(value):
    value = 0 if pd.isna(value) else float(value)
    return f"{value:.2f}%"


def num_fmt(value):
    value = 0 if pd.isna(value) else float(value)
    return f"{value:.2f}"


def safe_div(a, b):
    if b is None or b == 0 or pd.isna(b):
        return 0
    return a / b


def safe_div_series(a, b):
    return np.where(b == 0, 0, a / b)


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data(ttl=1800)
def load_data():
    sql = f"""
    SELECT
        market,
        type,
        diagonal_category,
        onkron_competitor,
        brand,
        asin,
        CAST(price AS DECIMAL(10,2)) AS price,
        CAST(revenue AS DECIMAL(14,2)) AS revenue,
        CAST(sales AS DECIMAL(12,2)) AS sales
    FROM amazon_competitors
    WHERE
        data_date_begin >= '{PERIOD_START}'
        AND data_date_begin < '{PERIOD_END}'
        AND type IS NOT NULL
        AND diagonal_category IS NOT NULL
    """

    df = run_query(sql)

    if df.empty:
        return df

    df["onkron_competitor"] = df["onkron_competitor"].fillna("").astype(str).str.lower()
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["type"] = df["type"].fillna("").astype(str)
    df["diagonal_category"] = df["diagonal_category"].fillna("").astype(str)
    df["market"] = df["market"].fillna("").astype(str)

    return df


# =========================================================
# SCORE LOGIC LIKE EXCEL
# =========================================================
def prepare_score(df):
    keys = ["market", "type", "diagonal_category"]

    grouped = (
        df.groupby(keys + ["onkron_competitor"], dropna=False)
        .agg(
            revenue=("revenue", "sum"),
            sales=("sales", "sum"),
            players=("asin", "nunique"),
            avg_price=("price", "mean"),
        )
        .reset_index()
    )

    competitors = grouped[grouped["onkron_competitor"] == "competitor"].copy()
    onkron = grouped[grouped["onkron_competitor"] == "onkron"].copy()

    comp = (
        competitors.groupby(keys, dropna=False)
        .agg(
            revenue=("revenue", "sum"),
            sales=("sales", "sum"),
            players=("players", "sum"),
            avg_price=("avg_price", "mean"),
        )
        .reset_index()
    )

    onk = (
        onkron.groupby(keys, dropna=False)
        .agg(
            onkron_revenue=("revenue", "sum"),
            onkron_sales_units=("sales", "sum"),
            onkron_models=("players", "sum"),
        )
        .reset_index()
    )

    score = comp.merge(onk, on=keys, how="left")

    for col in ["onkron_revenue", "onkron_sales_units", "onkron_models"]:
        score[col] = score[col].fillna(0)

    score["revenue_per_player"] = score["revenue"] / score["players"].replace(0, np.nan)
    score["revenue_per_player"] = score["revenue_per_player"].fillna(0)

    score["onkron_share"] = score["onkron_revenue"] / (
        score["revenue"] + score["onkron_revenue"] + 1
    )
    score["onkron_share_pct"] = score["onkron_share"] * 100

    score["opportunity_gap"] = 1 - score["onkron_share"]
    score["opportunity_gap_pct"] = score["opportunity_gap"] * 100

    max_revenue = score["revenue"].max()
    max_rev_player = score["revenue_per_player"].max()
    max_players = score["players"].max()

    score["revenue_norm"] = safe_div_series(score["revenue"], max_revenue)
    score["rev_player_norm"] = safe_div_series(score["revenue_per_player"], max_rev_player)
    score["low_competition_norm"] = 1 - safe_div_series(score["players"], max_players)

    score["score"] = (
        score["revenue_norm"] * (WEIGHT_REVENUE * 100)
        + score["rev_player_norm"] * (WEIGHT_REV_PLAYER * 100)
        + score["low_competition_norm"] * (WEIGHT_LOW_COMPETITION * 100)
        + score["opportunity_gap"] * (WEIGHT_OPPORTUNITY_GAP * 100)
    )

    score["priority"] = np.where(
        score["score"] >= THRESHOLD_HIGH,
        "HIGH",
        np.where(score["score"] >= THRESHOLD_MEDIUM, "MEDIUM", "LOW")
    )

    score["status"] = np.where(
        score["onkron_revenue"] > 0,
        "Текущий пайплайн",
        np.where(
            (score["revenue"] >= PIPELINE_MIN_REVENUE)
            & (score["onkron_share_pct"] <= PIPELINE_MAX_ONKRON_SHARE),
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

    return score.sort_values("score", ascending=False)


# =========================================================
# LOAD APP DATA
# =========================================================
st.title("Куда вводить продукт")
st.caption(f"Данные за {PERIOD_LABEL}")

try:
    df = load_data()
except Exception as e:
    st.error("Ошибка подключения к базе или загрузки данных.")
    st.exception(e)
    st.stop()

if df.empty:
    st.error("Нет данных за выбранный период.")
    st.stop()

score_df = prepare_score(df)


# =========================================================
# FILTERS
# =========================================================
markets = sorted(df["market"].dropna().unique())

col1, col2, col3 = st.columns(3)

with col1:
    selected_market = st.selectbox("Рынок", markets)

CUR = currency_symbol(selected_market)

filtered = df[df["market"] == selected_market].copy()
filtered_score = score_df[score_df["market"] == selected_market].copy()

with col2:
    type_options = ["Все"] + sorted(filtered["type"].dropna().unique())
    selected_type = st.selectbox("Type", type_options)

if selected_type != "Все":
    filtered = filtered[filtered["type"] == selected_type]
    filtered_score = filtered_score[filtered_score["type"] == selected_type]

with col3:
    diag_options = ["Все"] + sorted(filtered["diagonal_category"].dropna().unique())
    selected_diag = st.selectbox("Diagonal Category", diag_options)

if selected_diag != "Все":
    filtered = filtered[filtered["diagonal_category"] == selected_diag]
    filtered_score = filtered_score[filtered_score["diagonal_category"] == selected_diag]


# =========================================================
# KPI
# =========================================================
total_revenue = filtered["revenue"].sum()
onkron_revenue = filtered[filtered["onkron_competitor"] == "onkron"]["revenue"].sum()
onkron_share = safe_div(onkron_revenue, total_revenue + 1) * 100

kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric("Общий объем выручки", money_fmt(total_revenue, CUR))
kpi2.metric("Выручка ONKRON", money_fmt(onkron_revenue, CUR))
kpi3.metric("Доля ONKRON", pct_fmt(onkron_share))


# =========================================================
# TOP POTENTIAL PIPELINE
# =========================================================
st.subheader("Top Potential Pipeline")

pipeline = filtered_score[
    filtered_score["status"].isin(["Текущий пайплайн", "Потенциальный пайплайн"])
].copy()

if pipeline.empty:
    st.info("Нет сегментов для pipeline по текущим фильтрам.")
else:
    pipeline_show = pipeline[
        [
            "type",
            "diagonal_category",
            "revenue",
            "players",
            "revenue_per_player",
            "onkron_share_pct",
            "score",
            "priority",
        ]
    ].copy()

    pipeline_show.columns = [
        "Type",
        "Diagonal Category",
        "Выручка",
        "Кол-во игроков",
        "Выручка на игрока",
        "Доля ONKRON",
        "Score",
        "Priority",
    ]

    pipeline_show["Выручка"] = pipeline_show["Выручка"].apply(lambda x: money_fmt(x, CUR))
    pipeline_show["Кол-во игроков"] = pipeline_show["Кол-во игроков"].apply(int_fmt)
    pipeline_show["Выручка на игрока"] = pipeline_show["Выручка на игрока"].apply(lambda x: money_fmt(x, CUR))
    pipeline_show["Доля ONKRON"] = pipeline_show["Доля ONKRON"].apply(pct_fmt)
    pipeline_show["Score"] = pipeline_show["Score"].apply(num_fmt)

    st.dataframe(
        pipeline_show.head(20),
        use_container_width=True,
        height=360,
        hide_index=True,
    )


# =========================================================
# TOP SEGMENTS BY REVENUE
# =========================================================
left, right = st.columns([1, 1])

with left:
    st.subheader("Топ сегментов по выручке")

    top_segments = filtered_score.sort_values("revenue", ascending=False).head(10).copy()

    top_segments["segment"] = (
        top_segments["type"].astype(str)
        + " | "
        + top_segments["diagonal_category"].astype(str)
    )

    top_table = top_segments[["segment", "revenue", "score"]].copy()
    top_table.columns = ["Сегмент", "Revenue", "Score"]

    top_table["Revenue"] = top_table["Revenue"].apply(lambda x: money_fmt(x, CUR))
    top_table["Score"] = top_table["Score"].apply(num_fmt)

    st.dataframe(
        top_table,
        use_container_width=True,
        height=390,
        hide_index=True,
    )

with right:
    st.subheader("Top-10 segments by Revenue")

    if not top_segments.empty:
        fig = px.bar(
            top_segments.sort_values("revenue"),
            x="revenue",
            y="segment",
            orientation="h",
            text="revenue",
            labels={
                "revenue": f"Revenue, {CUR}",
                "segment": "Segment",
            },
        )

        if CUR == "RUB":
            fig.update_traces(
                texttemplate="%{text:,.2f} RUB",
                textposition="outside",
            )
        else:
            fig.update_traces(
                texttemplate=f"{CUR} %{{text:,.2f}}",
                textposition="outside",
            )

        fig.update_layout(
            height=390,
            margin=dict(l=10, r=100, t=20, b=40),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# MATRICES
# =========================================================
st.subheader("Матрицы Type × Diagonal Category")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("**ONKRON Share**")

    pivot_share = filtered_score.pivot_table(
        index="type",
        columns="diagonal_category",
        values="onkron_share_pct",
        aggfunc="sum",
        fill_value=0,
    )

    pivot_share_fmt = pivot_share.map(pct_fmt)

    st.dataframe(
        pivot_share_fmt,
        use_container_width=True,
        height=420,
    )

with m2:
    st.markdown("**Score**")

    pivot_score = filtered_score.pivot_table(
        index="type",
        columns="diagonal_category",
        values="score",
        aggfunc="mean",
        fill_value=0,
    )

    pivot_score_fmt = pivot_score.map(num_fmt)

    st.dataframe(
        pivot_score_fmt,
        use_container_width=True,
        height=420,
    )

with m3:
    st.markdown("**Revenue**")

    pivot_revenue = filtered_score.pivot_table(
        index="type",
        columns="diagonal_category",
        values="revenue",
        aggfunc="sum",
        fill_value=0,
    )

    pivot_revenue_fmt = pivot_revenue.map(lambda x: money_fmt(x, CUR))

    st.dataframe(
        pivot_revenue_fmt,
        use_container_width=True,
        height=420,
    )


# =========================================================
# MAIN SCORING TABLE
# =========================================================
st.subheader("Основная таблица скоринга")

table = filtered_score[
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
    "Revenue",
    "Sales",
    "Players(ASINs)",
    "Avg Price",
    "Revenue per player",
    "ONKRON Revenue",
    "ONKRON Sales units",
    "ONKRON Models",
    "ONKRON Share",
    "Opportunity Gap",
    "Revenue Norm",
    "Rev/Player Norm",
    "Low Competition Norm",
    "Score",
    "Приоритет",
    "Статус",
    "Рекомендации",
]

money_cols = [
    "Revenue",
    "Avg Price",
    "Revenue per player",
    "ONKRON Revenue",
]

int_cols = [
    "Sales",
    "Players(ASINs)",
    "ONKRON Sales units",
    "ONKRON Models",
]

pct_cols = [
    "ONKRON Share",
    "Opportunity Gap",
]

num_cols = [
    "Revenue Norm",
    "Rev/Player Norm",
    "Low Competition Norm",
    "Score",
]

for col in money_cols:
    table[col] = table[col].apply(lambda x: money_fmt(x, CUR))

for col in int_cols:
    table[col] = table[col].apply(int_fmt)

for col in pct_cols:
    table[col] = table[col].apply(pct_fmt)

for col in num_cols:
    table[col] = table[col].apply(num_fmt)

st.dataframe(
    table,
    use_container_width=True,
    height=620,
    hide_index=True,
)


# =========================================================
# SCORING SETTINGS
# =========================================================
st.subheader("Параметры автоматического скоринга")

settings = pd.DataFrame(
    [
        [
            "Вес емкости сегмента",
            "35%",
            "Revenue Norm",
            "Чем выше выручка сегмента, тем выше приоритет",
        ],
        [
            "Вес выручки на игрока",
            "25%",
            "Rev/Player Norm",
            "Показывает денежность сегмента на одного игрока/модель",
        ],
        [
            "Вес низкой конкуренции",
            "20%",
            "Low Competition Norm",
            "Больше балл, если игроков меньше",
        ],
        [
            "Вес разрыва по доле WE",
            "20%",
            "Opportunity Gap",
            "Больше балл, если наша доля низкая или нулевая",
        ],
        [
            "Порог HIGH",
            THRESHOLD_HIGH,
            "Score",
            "Приоритет HIGH, если Score >= порога",
        ],
        [
            "Порог MEDIUM",
            THRESHOLD_MEDIUM,
            "Score",
            "Приоритет MEDIUM, если Score >= порога",
        ],
        [
            "Порог минимальной выручки для Pipeline",
            money_fmt(PIPELINE_MIN_REVENUE, CUR),
            "Revenue",
            "Фильтр для потенциального pipeline",
        ],
        [
            "Порог максимальной доли WE для Pipeline",
            f"{PIPELINE_MAX_ONKRON_SHARE:.2f}%",
            "WE Share",
            "Сегмент считается потенциальным, если WE Share <= порога",
        ],
    ],
    columns=["Параметр", "Значение", "Используется в", "Комментарий"],
)

st.dataframe(
    settings,
    use_container_width=True,
    hide_index=True,
)