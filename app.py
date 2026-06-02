"""
Amazon Competitors PM Dashboard
================================
Дашборд для продуктового менеджера: куда вводить / откуда выводить продукт.
Данные: MySQL analyticallab → таблица amazon_competitors (через Onkron MCP).

Запуск:
    pip install pymysql pandas plotly dash python-dotenv --break-system-packages
    python amazon_pm_dashboard.py
    → http://127.0.0.1:8050
"""

import os
import math
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, dash_table
import pymysql
import sys
print(sys.executable)

# ──────────────────────────────────────────────
# 1. ПОДКЛЮЧЕНИЕ К БД
# ──────────────────────────────────────────────
DB_CONFIG = dict(
    host=os.getenv("DB_HOST", 'onkron.com'),
    port=int(os.getenv("DB_PORT", 3306)),
    user=os.getenv("DB_USER", "analyticallab"),
    password=os.getenv("DB_PASSWORD", "eL81h85tfZgIxYH"),
    database=os.getenv("DB_NAME", "analyticallab"),
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
)


def run_query(sql: str) -> pd.DataFrame:
    conn = pymysql.connect(**DB_CONFIG)

    try:
        with conn.cursor() as cur:
            cur.execute(sql)

            rows = cur.fetchall()

            if cur.description:
                columns = [col[0] for col in cur.description]
            else:
                columns = []

        return pd.DataFrame(rows, columns=columns)

    finally:
        conn.close()


# ──────────────────────────────────────────────
# 2. SQL-ЗАПРОСЫ
# ──────────────────────────────────────────────
test = run_query("""
SELECT
    COUNT(*) cnt,
    MIN(data_date_begin) min_date,
    MAX(data_date_begin) max_date
FROM amazon_competitors
""")

print(test)


SQL_SEGMENTS = """
SELECT
    market,
    type,
    diagonal_category,
    onkron_competitor,
    COUNT(DISTINCT asin)                        AS players,
    ROUND(SUM(revenue), 2)                      AS total_revenue,
    ROUND(AVG(CAST(price AS DECIMAL(10,2))), 2) AS avg_price,
    ROUND(AVG(CAST(rating AS DECIMAL(3,1))), 2) AS avg_rating,
    ROUND(AVG(CAST(reviews AS UNSIGNED)), 0)    AS avg_reviews,
    ROUND(SUM(CAST(sales AS DECIMAL(12,2))), 0) AS total_sales,
    ROUND(AVG(CAST(bsr AS DECIMAL(12,2))), 0)   AS avg_bsr
FROM amazon_competitors
WHERE
    data_date_begin >= '2026-04-01'
    AND data_date_begin < '2026-05-01'
    AND type IS NOT NULL
    AND diagonal_category IS NOT NULL
GROUP BY market, type, diagonal_category, onkron_competitor
"""

SQL_PRICE_DIST = """
-- Распределение цен по сегменту (для анализа ценовых ниш)
SELECT
    market,
    type,
    diagonal_category,
    onkron_competitor,
    CAST(price AS DECIMAL(10,2)) AS price,
    CAST(revenue AS DECIMAL(14,2)) AS revenue,
    CAST(rating  AS DECIMAL(3,1))  AS rating,
    CAST(reviews AS UNSIGNED)      AS reviews,
    CAST(sales   AS DECIMAL(12,2)) AS sales,
    brand,
    asin
FROM amazon_competitors
WHERE
    data_date_begin >= '2026-04-01'
    AND data_date_begin < '2026-05-01'
    AND price  IS NOT NULL
    AND type   IS NOT NULL
    AND diagonal_category IS NOT NULL
"""

SQL_TOP_BRANDS = """
-- Топ брендов-конкурентов по выручке
SELECT
    market,
    brand,
    ROUND(SUM(revenue),0) AS total_revenue,
    COUNT(DISTINCT asin)  AS skus,
    ROUND(AVG(CAST(rating AS DECIMAL(3,1))),2) AS avg_rating,
    ROUND(AVG(CAST(reviews AS UNSIGNED)),0) AS avg_reviews
FROM amazon_competitors
WHERE
    data_date_begin >= '2026-04-01'
    AND data_date_begin < '2026-05-01'
    AND onkron_competitor = 'competitor'
    AND brand IS NOT NULL
GROUP BY market, brand
ORDER BY total_revenue DESC
LIMIT 60
"""

SQL_BSR_RATING = """
-- BSR vs Rating (конкурентоспособность)
SELECT
    market,
    type,
    diagonal_category,
    onkron_competitor,
    brand,
    CAST(bsr    AS DECIMAL(12,2)) AS bsr,
    CAST(rating AS DECIMAL(3,1))  AS rating,
    CAST(reviews AS UNSIGNED)     AS reviews,
    CAST(revenue AS DECIMAL(14,2)) AS revenue,
    CAST(price  AS DECIMAL(10,2)) AS price
FROM amazon_competitors
WHERE
    data_date_begin >= '2026-04-01'
    AND data_date_begin < '2026-05-01'
    AND bsr    IS NOT NULL
    AND rating IS NOT NULL
    AND type   IS NOT NULL
    AND diagonal_category IS NOT NULL
"""


# ──────────────────────────────────────────────
# 3. ЗАГРУЗКА ДАННЫХ
# ──────────────────────────────────────────────
print("Загружаем данные из analyticallab...")
df_seg   = run_query(SQL_SEGMENTS)
df_price = run_query(SQL_PRICE_DIST)
df_brands= run_query(SQL_TOP_BRANDS)
df_bsr   = run_query(SQL_BSR_RATING)
print(f"  Сегменты: {len(df_seg)} строк")
print(f"  Цены:     {len(df_price)} строк")
print(f"  Бренды:   {len(df_brands)} строк")
print(f"  BSR:      {len(df_bsr)} строк")

print("df_seg columns:", list(df_seg.columns))
print("df_price columns:", list(df_price.columns))
print("df_brands columns:", list(df_brands.columns))
print("df_bsr columns:", list(df_bsr.columns))
# ──────────────────────────────────────────────
# 4. ВЫЧИСЛЯЕМЫЕ МЕТРИКИ
# ──────────────────────────────────────────────
def compute_opportunity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score = взвешенная оценка привлекательности сегмента для входа/выхода.
    Факторы:
      + высокая выручка сегмента          (нормализованная)
      + высокая цена (маржинальность)      (нормализованная)
      - большое число игроков (конкуренция)(инвертированная)
      + доля Onkron низкая (есть куда расти)
    Score 0–100. >65 → ENTER, <35 → EXIT, иначе WATCH.
    """
    comp = df[df["onkron_competitor"] == "competitor"].copy()
    onk  = df[df["onkron_competitor"] == "onkron"].copy()

    comp_agg = comp.groupby(["market","type","diagonal_category"]).agg(
        seg_revenue =("total_revenue","sum"),
        players      =("players","sum"),
        avg_price    =("avg_price","mean"),
        avg_rating   =("avg_rating","mean"),
    ).reset_index()

    onk_agg = onk.groupby(["market","type","diagonal_category"]).agg(
        onk_revenue=("total_revenue","sum"),
        onk_players=("players","sum"),
    ).reset_index()

    merged = comp_agg.merge(onk_agg, on=["market","type","diagonal_category"], how="left")
    merged["onk_revenue"]  = merged["onk_revenue"].fillna(0)
    merged["onk_players"]  = merged["onk_players"].fillna(0)
    merged["onkron_share"] = merged["onk_revenue"] / (merged["seg_revenue"] + merged["onk_revenue"] + 1)

    def norm(s, invert=False):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series([0.5]*len(s), index=s.index)
        n = (s - mn) / (mx - mn)
        return 1 - n if invert else n

    merged["s_revenue"]     = norm(merged["seg_revenue"])
    merged["s_price"]       = norm(merged["avg_price"])
    merged["s_competition"] = norm(merged["players"], invert=True)
    merged["s_gap"]         = norm(1 - merged["onkron_share"])

    merged["score"] = (
        merged["s_revenue"]     * 0.35 +
        merged["s_price"]       * 0.20 +
        merged["s_competition"] * 0.25 +
        merged["s_gap"]         * 0.20
    ) * 100

    merged["recommendation"] = merged["score"].apply(
        lambda x: "🟢 ENTER" if x >= 65 else ("🔴 EXIT" if x <= 35 else "🟡 WATCH")
    )
    return merged

if df_seg.empty:
    print("SQL_SEGMENTS вернул 0 строк")

    score_df = pd.DataFrame(
        columns=[
            "market",
            "type",
            "diagonal_category",
            "seg_revenue",
            "players",
            "avg_price",
            "avg_rating",
            "onk_revenue",
            "onk_players",
            "onkron_share",
            "score",
            "recommendation",
        ]
    )
else:
    score_df = compute_opportunity_score(df_seg)


# ──────────────────────────────────────────────
# 5. DASH APP
# ──────────────────────────────────────────────
MARKETS = sorted(df_price["market"].dropna().unique())

if not MARKETS:
    raise ValueError("Нет рынков в данных. Проверь SQL-запросы.")

COLORS = {
    "onkron":     "#0066CC",
    "competitor": "#E8443A",
    "enter":      "#27AE60",
    "exit":       "#E74C3C",
    "watch":      "#F39C12",
    "bg":         "#F4F6F9",
    "card":       "#FFFFFF",
    "text":       "#2C3E50",
    "accent":     "#2980B9",
}

app = dash.Dash(__name__, title="PM Dashboard · Amazon Wall Mounts")
app.layout = html.Div(style={"fontFamily":"Inter,Arial,sans-serif","backgroundColor":COLORS["bg"],"minHeight":"100vh","padding":"20px"}, children=[

    # ── Заголовок ──
    html.Div([
        html.H1("📊 Amazon Wall Mounts — PM Dashboard", style={"color":COLORS["text"],"margin":"0","fontSize":"24px"}),
        html.P("Куда вводить и откуда выводить продукт · Данные: апрель 2026", style={"color":"#7F8C8D","margin":"4px 0 0"}),
    ], style={"marginBottom":"20px"}),

    # ── Фильтры ──
    html.Div([
        html.Div([
            html.Label("Рынок", style={"fontWeight":"600","fontSize":"13px","color":COLORS["text"]}),
            dcc.Dropdown(id="dd-market", options=[{"label":m,"value":m} for m in MARKETS],
                         value=MARKETS[0], clearable=False, style={"fontSize":"13px"}),
        ], style={"width":"200px"}),
        html.Div([
            html.Label("Тип продукта", style={"fontWeight":"600","fontSize":"13px","color":COLORS["text"]}),
            dcc.Dropdown(id="dd-type", options=[], value=None, placeholder="Все",
                         style={"fontSize":"13px"}),
        ], style={"width":"260px","marginLeft":"16px"}),
    ], style={"display":"flex","alignItems":"flex-end","marginBottom":"20px"}),

    # ── KPI-карточки ──
    html.Div(id="kpi-cards", style={"display":"flex","gap":"12px","marginBottom":"20px","flexWrap":"wrap"}),

    # ── Строка 1: Opportunity Matrix + Score Table ──
    html.Div([
        html.Div([
            html.H3("🎯 Opportunity Matrix", style={"margin":"0 0 8px","fontSize":"15px","color":COLORS["text"]}),
            html.P("Пузырь = выручка сегмента. Ось X = конкуренция (игроков), Ось Y = Score привлекательности.",
                   style={"fontSize":"11px","color":"#95A5A6","margin":"0 0 8px"}),
            dcc.Graph(id="bubble-chart", style={"height":"380px"}),
        ], style={"flex":"1","background":COLORS["card"],"borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,.07)"}),

        html.Div([
            html.H3("📋 Рекомендации по сегментам", style={"margin":"0 0 8px","fontSize":"15px","color":COLORS["text"]}),
            dash_table.DataTable(
                id="score-table",
                columns=[
                    {"name":"Тип",       "id":"type"},
                    {"name":"Диагональ", "id":"diagonal_category"},
                    {"name":"Выручка €", "id":"seg_revenue","type":"numeric","format":{"specifier":",.0f"}},
                    {"name":"Игроков",   "id":"players","type":"numeric"},
                    {"name":"Ср. цена",  "id":"avg_price","type":"numeric","format":{"specifier":".2f"}},
                    {"name":"Score",     "id":"score","type":"numeric","format":{"specifier":".1f"}},
                    {"name":"Решение",   "id":"recommendation"},
                ],
                style_table={"height":"340px","overflowY":"auto"},
                style_cell={"fontSize":"12px","padding":"6px 10px","textAlign":"left","fontFamily":"Inter,Arial,sans-serif"},
                style_header={"backgroundColor":"#EBF5FB","fontWeight":"700","fontSize":"12px"},
                style_data_conditional=[
                    {"if":{"filter_query":'{recommendation} contains "ENTER"'},"backgroundColor":"#EAFAF1","color":"#1E8449"},
                    {"if":{"filter_query":'{recommendation} contains "EXIT"'},"backgroundColor":"#FDEDEC","color":"#C0392B"},
                    {"if":{"filter_query":'{recommendation} contains "WATCH"'},"backgroundColor":"#FEF9E7","color":"#B7770D"},
                ],
                sort_action="native",
                page_size=20,
            ),
        ], style={"width":"460px","background":COLORS["card"],"borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,.07)","marginLeft":"12px"}),
    ], style={"display":"flex","marginBottom":"12px"}),

    # ── Строка 2: Выручка по сегментам + Ценовые ниши ──
    html.Div([
        html.Div([
            html.H3("💰 Выручка: Onkron vs Конкуренты", style={"margin":"0 0 8px","fontSize":"15px","color":COLORS["text"]}),
            dcc.Graph(id="revenue-bar", style={"height":"330px"}),
        ], style={"flex":"1","background":COLORS["card"],"borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,.07)"}),

        html.Div([
            html.H3("💲 Ценовые ниши в сегменте", style={"margin":"0 0 8px","fontSize":"15px","color":COLORS["text"]}),
            html.P("Выберите тип и диагональ для детализации.", style={"fontSize":"11px","color":"#95A5A6","margin":"0 0 4px"}),
            dcc.Dropdown(id="dd-diag", options=[], value=None, placeholder="Диагональ",
                         style={"fontSize":"12px","marginBottom":"8px"}),
            dcc.Graph(id="price-box", style={"height":"280px"}),
        ], style={"width":"420px","background":COLORS["card"],"borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,.07)","marginLeft":"12px"}),
    ], style={"display":"flex","marginBottom":"12px"}),

    # ── Строка 3: BSR vs Rating + Топ брендов ──
    html.Div([
        html.Div([
            html.H3("⭐ BSR vs Рейтинг (конкурентная карта)", style={"margin":"0 0 8px","fontSize":"15px","color":COLORS["text"]}),
            html.P("Лучшая позиция: низкий BSR + высокий рейтинг (правый нижний угол).",
                   style={"fontSize":"11px","color":"#95A5A6","margin":"0 0 8px"}),
            dcc.Graph(id="bsr-scatter", style={"height":"340px"}),
        ], style={"flex":"1","background":COLORS["card"],"borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,.07)"}),

        html.Div([
            html.H3("🏆 Топ-15 конкурентов по выручке", style={"margin":"0 0 8px","fontSize":"15px","color":COLORS["text"]}),
            dcc.Graph(id="brands-bar", style={"height":"340px"}),
        ], style={"width":"420px","background":COLORS["card"],"borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,.07)","marginLeft":"12px"}),
    ], style={"display":"flex","marginBottom":"12px"}),

    # ── Строка 4: Heatmap (сегмент × диагональ) ──
    html.Div([
        html.Div([
            html.H3("🔥 Тепловая карта выручки: Тип × Диагональ", style={"margin":"0 0 8px","fontSize":"15px","color":COLORS["text"]}),
            html.P("Весь рынок (конкуренты + Onkron). Красный = высокая выручка → привлекательный сегмент.",
                   style={"fontSize":"11px","color":"#95A5A6","margin":"0 0 8px"}),
            dcc.Graph(id="heatmap", style={"height":"320px"}),
        ], style={"flex":"1","background":COLORS["card"],"borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,.07)"}),

        html.Div([
            html.H3("📈 Уровень проникновения Onkron", style={"margin":"0 0 8px","fontSize":"15px","color":COLORS["text"]}),
            html.P("Доля выручки Onkron от общей выручки сегмента. Низкая доля = пространство для роста.",
                   style={"fontSize":"11px","color":"#95A5A6","margin":"0 0 8px"}),
            dcc.Graph(id="share-chart", style={"height":"320px"}),
        ], style={"width":"440px","background":COLORS["card"],"borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,.07)","marginLeft":"12px"}),
    ], style={"display":"flex","marginBottom":"12px"}),

    # Примечание
    html.P("Источник: analyticallab → amazon_competitors | Период: апрель 2026 | Onkron MCP",
           style={"fontSize":"11px","color":"#BDC3C7","textAlign":"right","marginTop":"8px"}),
])


# ──────────────────────────────────────────────
# 6. CALLBACKS
# ──────────────────────────────────────────────

def card(label, value, color="#2980B9", subtitle=""):
    return html.Div([
        html.P(label, style={"margin":"0","fontSize":"11px","color":"#7F8C8D","fontWeight":"600","textTransform":"uppercase","letterSpacing":"0.5px"}),
        html.H2(value, style={"margin":"4px 0 0","fontSize":"22px","color":color,"fontWeight":"700"}),
        html.P(subtitle, style={"margin":"0","fontSize":"10px","color":"#95A5A6"}),
    ], style={"background":COLORS["card"],"borderRadius":"10px","padding":"14px 18px",
              "boxShadow":"0 2px 8px rgba(0,0,0,.07)","minWidth":"140px","borderTop":f"3px solid {color}"})


@app.callback(
    Output("dd-type","options"), Output("dd-type","value"),
    Input("dd-market","value")
)
def update_type_options(market):
    types = sorted(df_seg[df_seg["market"]==market]["type"].dropna().unique())
    opts = [{"label":t,"value":t} for t in types]
    return opts, None


@app.callback(
    Output("dd-diag","options"), Output("dd-diag","value"),
    Input("dd-market","value"), Input("dd-type","value")
)
def update_diag_options(market, ptype):
    sub = df_price[df_price["market"]==market]
    if ptype:
        sub = sub[sub["type"]==ptype]
    diags = sorted(sub["diagonal_category"].dropna().unique())
    opts = [{"label":d,"value":d} for d in diags]
    val = diags[0] if diags else None
    return opts, val


@app.callback(
    Output("kpi-cards","children"),
    Input("dd-market","value"), Input("dd-type","value")
)
def update_kpis(market, ptype):
    sub = df_seg[df_seg["market"]==market]
    if ptype:
        sub = sub[sub["type"]==ptype]

    total_rev  = sub["total_revenue"].sum()
    onk_rev    = sub[sub["onkron_competitor"]=="onkron"]["total_revenue"].sum()
    comp_rev   = sub[sub["onkron_competitor"]=="competitor"]["total_revenue"].sum()
    share      = onk_rev / (total_rev + 1) * 100
    n_players  = sub[sub["onkron_competitor"]=="competitor"]["players"].sum()
    avg_price_c= sub[sub["onkron_competitor"]=="competitor"]["avg_price"].mean()

    best_seg = score_df[score_df["market"]==market].nlargest(1,"score")
    best_label = ""
    if not best_seg.empty:
        r = best_seg.iloc[0]
        best_label = f"{r['type'][:20]} / {r['diagonal_category']}"

    return [
        card("Выручка рынка",     f"€{total_rev:,.0f}",   "#2980B9"),
        card("Выручка Onkron",    f"€{onk_rev:,.0f}",     "#27AE60"),
        card("Доля Onkron",       f"{share:.1f}%",         "#8E44AD" if share<5 else "#27AE60", "низкая → есть пространство" if share<5 else ""),
        card("Конкурентов (SKU)", f"{int(n_players)}",     "#E67E22"),
        card("Ср. цена конк.",    f"€{avg_price_c:.2f}",  "#16A085"),
        card("Топ сегмент",       best_label,              "#27AE60", "🟢 ENTER"),
    ]


@app.callback(
    Output("bubble-chart","figure"),
    Input("dd-market","value"), Input("dd-type","value")
)
def update_bubble(market, ptype):
    sub = score_df[score_df["market"]==market].copy()
    if ptype:
        sub = sub[sub["type"]==ptype]

    color_map = {"🟢 ENTER": COLORS["enter"], "🔴 EXIT": COLORS["exit"], "🟡 WATCH": COLORS["watch"]}
    fig = go.Figure()
    for rec, grp in sub.groupby("recommendation"):
        fig.add_trace(go.Scatter(
            x=grp["players"], y=grp["score"],
            mode="markers+text",
            marker=dict(
                size=[math.sqrt(max(r,1))/8 for r in grp["seg_revenue"]],
                color=color_map.get(rec,"gray"), opacity=0.75,
                line=dict(width=1,color="white"),
                sizemode="area", sizeref=0.5,
            ),
            text=[f"{t[:14]}<br>{d}" for t,d in zip(grp["type"],grp["diagonal_category"])],
            textposition="top center", textfont=dict(size=9),
            name=rec,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Игроков: %{x}<br>Score: %{y:.1f}<br>"
                "Выручка: €%{customdata:,.0f}<extra></extra>"
            ),
            customdata=grp["seg_revenue"],
        ))
    fig.add_hline(y=65, line_dash="dot", line_color=COLORS["enter"], annotation_text="Enter →")
    fig.add_hline(y=35, line_dash="dot", line_color=COLORS["exit"],  annotation_text="← Exit")
    fig.update_layout(
        xaxis_title="Кол-во конкурентов (SKU)", yaxis_title="Opportunity Score",
        yaxis=dict(range=[0,105]), margin=dict(l=40,r=10,t=10,b=40),
        legend=dict(orientation="h",yanchor="bottom",y=1,xanchor="right",x=1),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=11),
    )
    return fig


@app.callback(
    Output("score-table","data"),
    Input("dd-market","value"), Input("dd-type","value")
)
def update_table(market, ptype):
    sub = score_df[score_df["market"]==market].copy()
    if ptype:
        sub = sub[sub["type"]==ptype]
    sub = sub.sort_values("score", ascending=False)
    sub["seg_revenue"] = sub["seg_revenue"].round(0)
    sub["avg_price"]   = sub["avg_price"].round(2)
    sub["score"]       = sub["score"].round(1)
    return sub[["type","diagonal_category","seg_revenue","players","avg_price","score","recommendation"]].to_dict("records")


@app.callback(
    Output("revenue-bar","figure"),
    Input("dd-market","value"), Input("dd-type","value")
)
def update_revenue(market, ptype):
    sub = df_seg[df_seg["market"]==market].copy()
    if ptype:
        sub = sub[sub["type"]==ptype]

    sub["segment"] = sub["type"].str[:18] + " / " + sub["diagonal_category"]
    pivot = sub.groupby(["segment","onkron_competitor"])["total_revenue"].sum().reset_index()

    onk  = pivot[pivot["onkron_competitor"]=="onkron"].set_index("segment")["total_revenue"]
    comp = pivot[pivot["onkron_competitor"]=="competitor"].set_index("segment")["total_revenue"]
    segs = sorted(set(onk.index) | set(comp.index))

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Конкуренты", x=segs, y=[comp.get(s,0) for s in segs],
                         marker_color=COLORS["competitor"], opacity=0.8))
    fig.add_trace(go.Bar(name="ONKRON",     x=segs, y=[onk.get(s,0) for s in segs],
                         marker_color=COLORS["onkron"],     opacity=0.9))
    fig.update_layout(
        barmode="stack", xaxis_tickangle=-35,
        yaxis_title="Выручка €", margin=dict(l=40,r=10,t=10,b=90),
        legend=dict(orientation="h",y=1.05),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=11),
    )
    return fig


@app.callback(
    Output("price-box","figure"),
    Input("dd-market","value"), Input("dd-type","value"), Input("dd-diag","value")
)
def update_price_box(market, ptype, diag):
    sub = df_price[(df_price["market"]==market) & df_price["price"].notna()].copy()
    if ptype:
        sub = sub[sub["type"]==ptype]
    if diag:
        sub = sub[sub["diagonal_category"]==diag]
    if sub.empty:
        return go.Figure()

    sub["price"] = pd.to_numeric(sub["price"], errors="coerce")
    sub["label"] = sub["onkron_competitor"].map({"onkron":"ONKRON","competitor":"Конкуренты"})

    fig = px.violin(sub, x="label", y="price", color="label",
                    color_discrete_map={"ONKRON":COLORS["onkron"],"Конкуренты":COLORS["competitor"]},
                    box=True, points="outliers",
                    labels={"price":"Цена €","label":""})
    fig.update_layout(
        showlegend=False, margin=dict(l=30,r=10,t=10,b=30),
        yaxis_title="Цена €", plot_bgcolor="white", paper_bgcolor="white", font=dict(size=11),
    )
    return fig


@app.callback(
    Output("bsr-scatter","figure"),
    Input("dd-market","value"), Input("dd-type","value")
)
def update_bsr(market, ptype):
    sub = df_bsr[(df_bsr["market"]==market) & df_bsr["bsr"].notna() & df_bsr["rating"].notna()].copy()
    if ptype:
        sub = sub[sub["type"]==ptype]
    sub = sub.copy()
    sub["bsr"]    = pd.to_numeric(sub["bsr"], errors="coerce")
    sub["rating"] = pd.to_numeric(sub["rating"], errors="coerce")
    sub["revenue"]= pd.to_numeric(sub["revenue"], errors="coerce")
    sub = sub.dropna(subset=["bsr","rating"])
    sub = sub[sub["bsr"] < 200_000]  # убираем выбросы

    color_map = {"onkron":COLORS["onkron"],"competitor":COLORS["competitor"]}
    fig = px.scatter(sub, x="bsr", y="rating",
                     color="onkron_competitor",
                     color_discrete_map=color_map,
                     size="revenue", size_max=20,
                     hover_data=["brand","type","diagonal_category","price"],
                     labels={"bsr":"BSR (чем меньше — тем лучше)","rating":"Рейтинг ⭐","onkron_competitor":""},
                     opacity=0.65)
    # Добавляем квадранты
    med_bsr = sub["bsr"].median()
    med_rat = sub["rating"].median()
    fig.add_vline(x=med_bsr, line_dash="dash", line_color="gray", opacity=0.4)
    fig.add_hline(y=med_rat, line_dash="dash", line_color="gray", opacity=0.4)
    fig.add_annotation(x=med_bsr*0.5, y=med_rat*1.01, text="🏆 Лидеры", showarrow=False, font=dict(size=10,color="#27AE60"))
    fig.add_annotation(x=med_bsr*1.5, y=sub["rating"].min()*1.02, text="⚠️ Слабые", showarrow=False, font=dict(size=10,color="#E74C3C"))
    fig.update_layout(
        margin=dict(l=40,r=10,t=10,b=40),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=11),
        legend=dict(orientation="h",y=1.05),
    )
    return fig


@app.callback(
    Output("brands-bar","figure"),
    Input("dd-market","value"), Input("dd-type","value")
)
def update_brands(market, ptype):
    sub = df_brands[df_brands["market"]==market].copy()
    top = sub.nlargest(15,"total_revenue")
    colors_list = [COLORS["onkron"] if b=="ONKRON" else COLORS["competitor"] for b in top["brand"]]
    fig = go.Figure(go.Bar(
        x=top["total_revenue"], y=top["brand"],
        orientation="h",
        marker_color=colors_list,
        text=[f"€{v:,.0f}" for v in top["total_revenue"]],
        textposition="outside", textfont=dict(size=10),
    ))
    fig.update_layout(
        xaxis_title="Выручка €", yaxis=dict(autorange="reversed"),
        margin=dict(l=10,r=60,t=10,b=40),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=11),
    )
    return fig


@app.callback(
    Output("heatmap","figure"),
    Input("dd-market","value")
)
def update_heatmap(market):
    sub = df_seg[df_seg["market"]==market].groupby(["type","diagonal_category"])["total_revenue"].sum().reset_index()
    pivot = sub.pivot(index="type", columns="diagonal_category", values="total_revenue").fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale="RdYlGn", text=[[f"€{v:,.0f}" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Выручка: %{text}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Диагональ", yaxis_title="",
        margin=dict(l=10,r=10,t=10,b=40),
        paper_bgcolor="white", font=dict(size=11),
    )
    return fig


@app.callback(
    Output("share-chart","figure"),
    Input("dd-market","value")
)
def update_share(market):
    sub = df_seg[df_seg["market"]==market].copy()
    onk  = sub[sub["onkron_competitor"]=="onkron"].groupby("type")["total_revenue"].sum()
    comp = sub[sub["onkron_competitor"]=="competitor"].groupby("type")["total_revenue"].sum()
    types = sorted(set(onk.index)|set(comp.index))
    shares = [(onk.get(t,0)/(onk.get(t,0)+comp.get(t,0)+1))*100 for t in types]

    bar_colors = [COLORS["enter"] if s<10 else (COLORS["watch"] if s<30 else COLORS["onkron"]) for s in shares]
    fig = go.Figure(go.Bar(
        x=[t[:22] for t in types], y=shares,
        marker_color=bar_colors,
        text=[f"{s:.1f}%" for s in shares],
        textposition="outside",
    ))
    fig.add_hline(y=10, line_dash="dot", line_color=COLORS["enter"], annotation_text="10%")
    fig.update_layout(
        yaxis_title="Доля Onkron %", xaxis_tickangle=-35,
        margin=dict(l=10,r=10,t=10,b=90),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=11),
    )
    return fig


# ──────────────────────────────────────────────
# 7. ЗАПУСК
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)