import os
import numpy as np
import pandas as pd
import pymysql
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Анализ рынка",
    layout="wide"
)


# =========================================================
# SETTINGS
# =========================================================
PERIOD_LABELS = {
    "Март 2026": ("2026-03-01", "2026-04-01"),
    "Апрель 2026": ("2026-04-01", "2026-05-01"),
}

PERIOD_DEFAULT = "Апрель 2026"

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

    if (
        "uk" in market
        or "co.uk" in market
        or "united kingdom" in market
        or "britain" in market
        or "amazon uk" in market
    ):
        return "£"

    if (
        "usa" in market
        or market == "us"
        or "amazon.com" in market
        or ".com" in market
    ):
        return "$"

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

    return "€"


def money_fmt(value, symbol):
    value = 0 if pd.isna(value) else float(value)

    if symbol == "RUB":
        return f"{value:,.2f} RUB"

    return f"{symbol} {value:,.2f}"


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


def get_prev_period(period_name):
    if period_name == "Апрель 2026":
        return "2026-03-01", "2026-04-01"

    return None, None


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data(ttl=1800)
def load_data():
    sql = """
    SELECT
        onkron_category,
        market,
        type,
        diagonal_category,
        load_capacity_kg_category,
        onkron_competitor,
        brand,
        asin,
        data_date_begin,
        CAST(price AS DECIMAL(10,2)) AS price,
        CAST(revenue AS DECIMAL(14,2)) AS revenue,
        CAST(sales AS DECIMAL(12,2)) AS sales
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

    df["onkron_category"] = (
        df["onkron_category"]
        .fillna("Без категории")
        .astype(str)
        .str.strip()
    )
    df["onkron_category"] = df["onkron_category"].replace("", "Без категории")

    df["onkron_competitor"] = (
        df["onkron_competitor"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["brand"] = df["brand"].fillna("Без бренда").astype(str).str.strip()
    df["brand"] = df["brand"].replace("", "Без бренда")

    df["type"] = df["type"].fillna("Без типа").astype(str).str.strip()
    df["type"] = df["type"].replace("", "Без типа")

    df["diagonal_category"] = (
        df["diagonal_category"]
        .fillna("Без диагонали")
        .astype(str)
        .str.strip()
    )
    df["diagonal_category"] = df["diagonal_category"].replace("", "Без диагонали")

    df["load_capacity_kg_category"] = (
        df["load_capacity_kg_category"]
        .fillna("Без нагрузки")
        .astype(str)
        .str.strip()
    )
    df["load_capacity_kg_category"] = df["load_capacity_kg_category"].replace("", "Без нагрузки")

    df["market"] = df["market"].fillna("").astype(str).str.strip()

    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["data_date_begin"] = pd.to_datetime(df["data_date_begin"], errors="coerce")

    return df


# =========================================================
# SCORE LOGIC
# =========================================================
def prepare_score(df):
    keys = [
        "market",
        "type",
        "diagonal_category",
        "load_capacity_kg_category",
    ]

    if df.empty:
        return pd.DataFrame()

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
        "Высокий",
        np.where(score["score"] >= THRESHOLD_MEDIUM, "Средний", "Низкий")
    )

    score["status"] = np.where(
        score["onkron_revenue"] > 0,
        "Текущий ассортимент",
        np.where(
            (score["revenue"] >= PIPELINE_MIN_REVENUE)
            & (score["onkron_share_pct"] <= PIPELINE_MAX_ONKRON_SHARE),
            "Потенциальная возможность",
            "Низкий приоритет"
        )
    )

    score["recommendation"] = np.where(
        score["status"] == "Текущий ассортимент",
        "Масштабировать / защищать позицию",
        np.where(
            score["status"] == "Потенциальная возможность",
            "Оценить запуск / закрыть ассортиментный gap",
            "Пока не приоритет"
        )
    )

    return score.sort_values("score", ascending=False)


# =========================================================
# TABLE HELPER
# =========================================================
def priority_style(value):
    if value == "Высокий":
        return "background-color: #c6efce; color: #006100; font-weight: 700;"
    if value == "Средний":
        return "background-color: #ffeb9c; color: #9c6500; font-weight: 700;"
    if value == "Низкий":
        return "background-color: #ffc7ce; color: #9c0006; font-weight: 700;"
    return ""

def show_market_dynamics(df, selected_market, selected_category, cur):
    st.subheader("Динамика рынка")

    dynamics = df[df["market"] == selected_market].copy()

    if selected_category != "Все категории":
        dynamics = dynamics[dynamics["onkron_category"] == selected_category].copy()

    if dynamics.empty:
        st.info("Нет данных для динамики рынка.")
        return

    dynamics["month"] = dynamics["data_date_begin"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        dynamics.groupby("month", dropna=False)
        .agg(
            market_revenue=("revenue", "sum"),
            onkron_revenue=(
                "revenue",
                lambda x: x[dynamics.loc[x.index, "onkron_competitor"] == "onkron"].sum()
            ),
        )
        .reset_index()
        .sort_values("month")
    )

    monthly["month_label"] = monthly["month"].dt.strftime("%m.%Y")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly["month_label"],
            y=monthly["market_revenue"],
            mode="lines+markers+text",
            name="Объем рынка",
            text=monthly["market_revenue"],
            texttemplate=(
                "%{text:,.0f} RUB"
                if cur == "RUB"
                else f"{cur} %{{text:,.0f}}"
            ),
            textposition="top center",
            hovertemplate=(
                "<b>%{x}</b><br>"
                + (
                    "Объем рынка: %{y:,.2f} RUB"
                    if cur == "RUB"
                    else f"Объем рынка: {cur} %{{y:,.2f}}"
                )
                + "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly["month_label"],
            y=monthly["onkron_revenue"],
            mode="lines+markers+text",
            name="Объем ONKRON",
            text=monthly["onkron_revenue"],
            texttemplate=(
                "%{text:,.0f} RUB"
                if cur == "RUB"
                else f"{cur} %{{text:,.0f}}"
            ),
            textposition="bottom center",
            hovertemplate=(
                "<b>%{x}</b><br>"
                + (
                    "Объем ONKRON: %{y:,.2f} RUB"
                    if cur == "RUB"
                    else f"Объем ONKRON: {cur} %{{y:,.2f}}"
                )
                + "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            title="Месяц",
            fixedrange=False,
        ),
        yaxis=dict(
            title=f"Выручка, {cur}",
            fixedrange=False,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def show_scoring_table(filtered_score, cur):
    st.subheader("Основная таблица скоринга")

    if filtered_score.empty:
        st.info("Нет данных для таблицы скоринга.")
        return

    table = filtered_score[
        [
            "type",
            "diagonal_category",
            "load_capacity_kg_category",
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
        "Тип продукта",
        "Диагональ",
        "Нагрузка",
        "Выручка рынка",
        "Продажи, шт.",
        "Количество ASIN",
        "Средняя цена",
        "Выручка на ASIN",
        "Выручка ONKRON",
        "Продажи ONKRON, шт.",
        "Модели ONKRON",
        "Доля ONKRON",
        "Разрыв по доле ONKRON",
        "Индекс выручки",
        "Индекс выручки на ASIN",
        "Индекс низкой конкуренции",
        "Score",
        "Приоритет",
        "Статус",
        "Рекомендация",
    ]

    money_cols = [
        "Выручка рынка",
        "Средняя цена",
        "Выручка на ASIN",
        "Выручка ONKRON",
    ]

    int_cols = [
        "Продажи, шт.",
        "Количество ASIN",
        "Продажи ONKRON, шт.",
        "Модели ONKRON",
    ]

    pct_cols = [
        "Доля ONKRON",
        "Разрыв по доле ONKRON",
    ]

    num_cols = [
        "Индекс выручки",
        "Индекс выручки на ASIN",
        "Индекс низкой конкуренции",
        "Score",
    ]

    for col in money_cols:
        table[col] = table[col].apply(lambda x: money_fmt(x, cur))

    for col in int_cols:
        table[col] = table[col].apply(int_fmt)

    for col in pct_cols:
        table[col] = table[col].apply(pct_fmt)

    for col in num_cols:
        table[col] = table[col].apply(num_fmt)

    styled_table = table.style.map(priority_style, subset=["Приоритет"])

    st.dataframe(
        styled_table,
        use_container_width=True,
        height=620,
        hide_index=True,
    )


# =========================================================
# HEATMAP HELPER
# =========================================================
def make_heatmap(data, value_col, title, text_func, color_scale="YlGn"):
    pivot = data.pivot_table(
        index="type",
        columns="diagonal_category",
        values=value_col,
        aggfunc="sum" if value_col != "score" else "mean",
        fill_value=0,
    )

    if pivot.empty:
        st.info(f"Нет данных для: {title}")
        return

    text_values = pivot.map(text_func)

    fig = px.imshow(
        pivot,
        text_auto=False,
        aspect="auto",
        color_continuous_scale=color_scale,
        labels=dict(
            x="Диагональ",
            y="Тип продукта",
            color=title,
        ),
    )

    fig.update_traces(
        text=text_values.values,
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{text}<extra></extra>",
    )

    fig.update_layout(
        title=title,
        height=max(520, len(pivot.index) * 32),
        margin=dict(l=10, r=10, t=50, b=40),
        xaxis=dict(side="top", automargin=True),
        yaxis=dict(automargin=True),
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# BRAND CHART HELPER
# =========================================================
def show_brand_chart(filtered, cur):
    st.subheader("Бренды: выручка и доля рынка")

    brand_df = (
        filtered.groupby("brand", dropna=False)
        .agg(revenue=("revenue", "sum"))
        .reset_index()
    )

    brand_df["brand"] = (
        brand_df["brand"]
        .fillna("Без бренда")
        .astype(str)
        .str.strip()
    )

    brand_df = brand_df[brand_df["revenue"] > 0].copy()

    if brand_df.empty:
        st.info("Нет данных по брендам.")
        return

    brand_df = brand_df.sort_values("revenue", ascending=False)

    total_revenue = brand_df["revenue"].sum()
    brand_df["market_share_pct"] = brand_df["revenue"] / total_revenue * 100

    brand_df["label_for_chart"] = np.where(
        brand_df["market_share_pct"] >= 3,
        brand_df["brand"],
        ""
    )

    brand_df["pull"] = np.where(
        brand_df["brand"].str.upper().str.contains("ONKRON", na=False),
        0.08,
        0
    )

    brand_df["color"] = np.where(
        brand_df["brand"].str.upper().str.contains("ONKRON", na=False),
        "#00C2C7",
        "#D9D9D9"
    )

    fig = go.Figure(
        data=[
            go.Pie(
                labels=brand_df["brand"],
                values=brand_df["revenue"],
                hole=0.45,
                pull=brand_df["pull"],
                marker=dict(
                    colors=brand_df["color"],
                    line=dict(color="white", width=1)
                ),
                text=brand_df["label_for_chart"],
                textinfo="text+percent",
                textposition="inside",
                insidetextorientation="radial",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    + (
                        "Выручка: %{value:,.2f} RUB<br>"
                        if cur == "RUB"
                        else f"Выручка: {cur} %{{value:,.2f}}<br>"
                    )
                    + "Доля рынка: %{percent}"
                    + "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_traces(
        sort=False,
        showlegend=True
    )

    fig.update_layout(
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Все бренды отображены на диаграмме. Подписи скрываются у маленьких долей, но доступны при наведении мыши."
    )


# =========================================================
# LOAD APP DATA
# =========================================================
try:
    df = load_data()
except Exception as e:
    st.error("Ошибка подключения к базе или загрузки данных.")
    st.exception(e)
    st.stop()

if df.empty:
    st.error("Нет данных за выбранный период.")
    st.stop()


# =========================================================
# HEADER + CATEGORY FILTER
# =========================================================
categories = ["Все категории"] + sorted(df["onkron_category"].dropna().unique())

header_col, category_col = st.columns([3, 1])

with header_col:
    st.title("Анализ рынка")

with category_col:
    selected_category = st.selectbox("Категория", categories)


# =========================================================
# FILTERS
# =========================================================
markets = sorted(df["market"].dropna().unique())

col1, col2 = st.columns(2)

with col1:
    selected_market = st.selectbox("Рынок", markets)

with col2:
    selected_period = st.selectbox(
        "Период",
        list(PERIOD_LABELS.keys()),
        index=list(PERIOD_LABELS.keys()).index(PERIOD_DEFAULT),
    )

CUR = currency_symbol(selected_market)

selected_start, selected_end = PERIOD_LABELS[selected_period]
prev_start, prev_end = get_prev_period(selected_period)

selected_start_dt = pd.to_datetime(selected_start)
selected_end_dt = pd.to_datetime(selected_end)

filtered = df[
    (df["market"] == selected_market)
    & (df["data_date_begin"] >= selected_start_dt)
    & (df["data_date_begin"] < selected_end_dt)
].copy()

if selected_category != "Все категории":
    filtered = filtered[filtered["onkron_category"] == selected_category].copy()

if prev_start and prev_end:
    prev_start_dt = pd.to_datetime(prev_start)
    prev_end_dt = pd.to_datetime(prev_end)

    prev_filtered = df[
        (df["market"] == selected_market)
        & (df["data_date_begin"] >= prev_start_dt)
        & (df["data_date_begin"] < prev_end_dt)
    ].copy()

    if selected_category != "Все категории":
        prev_filtered = prev_filtered[
            prev_filtered["onkron_category"] == selected_category
        ].copy()
else:
    prev_filtered = pd.DataFrame(columns=df.columns)

if filtered.empty:
    st.warning("Нет данных по выбранным фильтрам.")
    st.stop()

filtered_score = prepare_score(filtered)

st.caption(f"Данные за период: {selected_period}")


# =========================================================
# KPI
# =========================================================
total_revenue = filtered["revenue"].sum()
onkron_revenue = filtered[filtered["onkron_competitor"] == "onkron"]["revenue"].sum()
onkron_share = safe_div(onkron_revenue, total_revenue + 1) * 100

prev_total_revenue = prev_filtered["revenue"].sum() if not prev_filtered.empty else 0
prev_onkron_revenue = (
    prev_filtered[prev_filtered["onkron_competitor"] == "onkron"]["revenue"].sum()
    if not prev_filtered.empty
    else 0
)
prev_onkron_share = safe_div(prev_onkron_revenue, prev_total_revenue + 1) * 100

market_volume_delta_abs = total_revenue - prev_total_revenue
market_volume_delta_pct = (
    safe_div(market_volume_delta_abs, prev_total_revenue) * 100
    if prev_total_revenue > 0
    else 0
)

if selected_period == "Март 2026":
    onkron_share_mom_pct = 0
else:
    onkron_share_mom_pct = (
        safe_div(onkron_share - prev_onkron_share, prev_onkron_share) * 100
        if prev_onkron_share > 0
        else 0
    )

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "Объем рынка",
    money_fmt(total_revenue, CUR),
    delta=money_fmt(market_volume_delta_abs, CUR),
)

kpi2.metric(
    "Изменение объема рынка",
    f"{market_volume_delta_pct:.2f}%",
    delta=f"{market_volume_delta_pct:.2f}%",
)

kpi3.metric(
    "Выручка ONKRON",
    money_fmt(onkron_revenue, CUR),
)

kpi4.metric(
    "Доля ONKRON",
    pct_fmt(onkron_share),
    delta=f"{onkron_share_mom_pct:.2f}% MoM",
)

# =========================================================
# MARKET DYNAMICS
# =========================================================
show_market_dynamics(df, selected_market, selected_category, CUR)


# =========================================================
# MAIN SCORING TABLE
# =========================================================
show_scoring_table(filtered_score, CUR)


# =========================================================
# TOP SEGMENTS BY REVENUE
# =========================================================
st.subheader("Топ-10 сегментов по выручке")

top_segments = filtered_score.sort_values("revenue", ascending=False).head(10).copy()

if top_segments.empty:
    st.info("Нет данных по сегментам.")
else:
    top_segments["segment"] = (
        top_segments["type"].astype(str)
        + " | "
        + top_segments["diagonal_category"].astype(str)
        + " | "
        + top_segments["load_capacity_kg_category"].astype(str)
    )

    fig = px.bar(
        top_segments.sort_values("revenue"),
        x="revenue",
        y="segment",
        orientation="h",
        text="revenue",
        labels={
            "revenue": f"Выручка, {CUR}",
            "segment": "Сегмент",
        },
    )

    if CUR == "RUB":
        fig.update_traces(
            texttemplate="%{text:,.2f} RUB",
            textposition="outside",
            cliponaxis=False,
        )
    else:
        fig.update_traces(
            texttemplate=f"{CUR} %{{text:,.2f}}",
            textposition="outside",
            cliponaxis=False,
        )

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=140, t=20, b=40),
        showlegend=False,
        yaxis=dict(automargin=True),
        xaxis=dict(fixedrange=False),
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# HEATMAP MATRICES
# =========================================================
st.subheader("Матрицы: тип продукта × диагональ")

tab1, tab2, tab3 = st.tabs([
    "Доля ONKRON",
    "Итоговый score",
    "Выручка рынка",
])

with tab1:
    make_heatmap(
        filtered_score,
        value_col="onkron_share_pct",
        title="Доля ONKRON",
        text_func=pct_fmt,
        color_scale="YlGn",
    )

with tab2:
    make_heatmap(
        filtered_score,
        value_col="score",
        title="Итоговый score",
        text_func=num_fmt,
        color_scale="YlGn",
    )

with tab3:
    make_heatmap(
        filtered_score,
        value_col="revenue",
        title="Выручка рынка",
        text_func=lambda x: money_fmt(x, CUR),
        color_scale="YlOrBr",
    )


# =========================================================
# BRAND REVENUE AND MARKET SHARE CHART
# =========================================================
show_brand_chart(filtered, CUR)


# =========================================================
# SCORING SETTINGS — BOTTOM OF PAGE
# =========================================================
st.subheader("Параметры автоматического скоринга")

settings = pd.DataFrame(
    [
        [
            "Вес емкости сегмента",
            "35%",
            "Индекс выручки",
            "Чем выше выручка сегмента, тем выше приоритет",
        ],
        [
            "Вес выручки на ASIN",
            "25%",
            "Индекс выручки на ASIN",
            "Показывает денежность сегмента на одну модель / ASIN",
        ],
        [
            "Вес низкой конкуренции",
            "20%",
            "Индекс низкой конкуренции",
            "Чем меньше игроков в сегменте, тем выше балл",
        ],
        [
            "Вес разрыва по доле ONKRON",
            "20%",
            "Разрыв по доле ONKRON",
            "Чем ниже текущая доля ONKRON, тем выше потенциал роста",
        ],
        [
            "Порог высокого приоритета",
            THRESHOLD_HIGH,
            "Score",
            "Сегмент получает высокий приоритет, если Score не ниже этого значения",
        ],
        [
            "Порог среднего приоритета",
            THRESHOLD_MEDIUM,
            "Score",
            "Сегмент получает средний приоритет, если Score не ниже этого значения",
        ],
        [
            "Порог минимальной выручки для Pipeline",
            money_fmt(PIPELINE_MIN_REVENUE, CUR),
            "Выручка рынка",
            "Минимальный размер сегмента для попадания в потенциальные возможности",
        ],
        [
            "Порог максимальной доли ONKRON для Pipeline",
            f"{PIPELINE_MAX_ONKRON_SHARE:.2f}%",
            "ONKRON Share",
            "Сегмент считается потенциальной возможностью, если доля ONKRON не выше порога",
        ],
    ],
    columns=["Параметр", "Значение", "Используется в", "Комментарий"],
)

st.dataframe(
    settings,
    use_container_width=True,
    hide_index=True,
)