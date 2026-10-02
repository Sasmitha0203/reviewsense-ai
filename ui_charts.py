"""
Plotly Chart Builders for Product Review Sentiment Analysis.
Features modern dark mode theme, sleek tooltips, and responsive layout.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List

COLOR_POSITIVE = "#10b981"  # Emerald
COLOR_NEGATIVE = "#f43f5e"  # Rose/Coral
COLOR_NEUTRAL  = "#f59e0b"  # Amber
COLOR_BG       = "rgba(0,0,0,0)"
COLOR_TEXT     = "#e2e8f0"
COLOR_GRID     = "rgba(255, 255, 255, 0.08)"

def create_gauge_chart(score: float, title: str = "Sentiment Intensity") -> go.Figure:
    """Creates a sleek semi-circular gauge meter from -1.0 (Very Negative) to +1.0 (Very Positive)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': COLOR_TEXT, 'family': 'Outfit'}},
        number={'font': {'size': 32, 'color': '#ffffff', 'family': 'Outfit'}, 'valueformat': "+.2f"},
        gauge={
            'axis': {'range': [-1.0, 1.0], 'tickwidth': 1, 'tickcolor': COLOR_TEXT, 'tickfont': {'color': '#94a3b8'}},
            'bar': {'color': COLOR_POSITIVE if score > 0.05 else (COLOR_NEGATIVE if score < -0.05 else COLOR_NEUTRAL), 'thickness': 0.28},
            'bgcolor': "rgba(255,255,255,0.04)",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.12)",
            'steps': [
                {'range': [-1.0, -0.05], 'color': 'rgba(244, 63, 94, 0.18)'},
                {'range': [-0.05, 0.05], 'color': 'rgba(245, 158, 11, 0.18)'},
                {'range': [0.05, 1.0], 'color': 'rgba(16, 185, 129, 0.18)'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font={'color': COLOR_TEXT, 'family': 'Plus Jakarta Sans'},
        height=240,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig


def create_sentiment_donut(counts: Dict[str, int]) -> go.Figure:
    """Creates an interactive donut chart of sentiment distribution."""
    labels = list(counts.keys())
    values = list(counts.values())

    color_map = {
        "Positive": COLOR_POSITIVE,
        "Neutral": COLOR_NEUTRAL,
        "Negative": COLOR_NEGATIVE
    }
    colors = [color_map.get(lbl, "#6366f1") for lbl in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        hoverinfo="label+percent+value",
        textinfo="percent",
        textfont=dict(size=14, color="#ffffff", family="Outfit")
    )])

    total = sum(values)
    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT, family="Plus Jakarta Sans"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color=COLOR_TEXT)
        ),
        annotations=[dict(
            text=f"<b>{total}</b><br><span style='font-size:11px;color:#94a3b8;'>Reviews</span>",
            x=0.5, y=0.5,
            font_size=20,
            font_family="Outfit",
            font_color="#ffffff",
            showarrow=False
        )],
        height=280,
        margin=dict(l=10, r=10, t=10, b=30)
    )
    return fig


def create_aspect_breakdown_chart(aspect_dict: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Horizontal stacked bar chart showing Aspect breakdown by sentiment percentage."""
    aspects = []
    pos_pcts = []
    neg_pcts = []
    neu_pcts = []

    for aspect, data in aspect_dict.items():
        total = data["total"]
        if total > 0:
            aspects.append(aspect)
            pos_pcts.append(round((data["pos"] / total) * 100, 1))
            neg_pcts.append(round((data["neg"] / total) * 100, 1))
            neu_pcts.append(round(((total - data["pos"] - data["neg"]) / total) * 100, 1))

    if not aspects:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=aspects, x=pos_pcts, name="Positive %", orientation='h',
        marker=dict(color=COLOR_POSITIVE, line=dict(color="rgba(255,255,255,0.1)", width=1))
    ))
    fig.add_trace(go.Bar(
        y=aspects, x=neu_pcts, name="Neutral %", orientation='h',
        marker=dict(color=COLOR_NEUTRAL, line=dict(color="rgba(255,255,255,0.1)", width=1))
    ))
    fig.add_trace(go.Bar(
        y=aspects, x=neg_pcts, name="Negative %", orientation='h',
        marker=dict(color=COLOR_NEGATIVE, line=dict(color="rgba(255,255,255,0.1)", width=1))
    ))

    fig.update_layout(
        barmode='stack',
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT, family="Plus Jakarta Sans"),
        xaxis=dict(
            title="Percentage (%)",
            range=[0, 100],
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12, color="#cbd5e1", family="Plus Jakarta Sans")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ),
        height=320,
        margin=dict(l=10, r=10, t=30, b=40)
    )
    return fig


def create_rating_sentiment_scatter(df: pd.DataFrame, rating_col: str, score_col: str) -> go.Figure:
    """Visualizes correlation between Customer Star Rating and Sentiment Score."""
    fig = px.strip(
        df,
        x=rating_col,
        y=score_col,
        color="sentiment_label",
        color_discrete_map={"Positive": COLOR_POSITIVE, "Neutral": COLOR_NEUTRAL, "Negative": COLOR_NEGATIVE},
        hover_data=["sentiment_label", score_col] if "review_text" not in df.columns else ["review_text"],
        stripmode='overlay'
    )

    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT, family="Plus Jakarta Sans"),
        xaxis=dict(
            title="Customer Star Rating (1 - 5)",
            tickmode='linear',
            tick0=1,
            dtick=1,
            gridcolor=COLOR_GRID
        ),
        yaxis=dict(
            title="Sentiment Polarity (-1.0 to +1.0)",
            range=[-1.05, 1.05],
            gridcolor=COLOR_GRID,
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        legend=dict(title="Sentiment", orientation="h", y=1.08, x=1, xanchor="right"),
        height=320,
        margin=dict(l=10, r=10, t=30, b=40)
    )
    return fig


def create_emotion_bar_chart(emotions: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart for emotion breakdown."""
    emos = list(emotions.keys())
    vals = list(emotions.values())

    colors = []
    for emo in emos:
        if "Joy" in emo:
            colors.append("#10b981")
        elif "Admiration" in emo:
            colors.append("#06b6d4")
        elif "Frustration" in emo:
            colors.append("#f43f5e")
        elif "Disappointment" in emo:
            colors.append("#f97316")
        else:
            colors.append("#f59e0b")

    fig = go.Figure(go.Bar(
        x=vals,
        y=emos,
        orientation='h',
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{v:.1f}%" for v in vals],
        textposition="auto",
        textfont=dict(color="#ffffff", family="Outfit")
    ))

    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT, family="Plus Jakarta Sans"),
        xaxis=dict(title="Percentage (%)", range=[0, 100], gridcolor=COLOR_GRID),
        yaxis=dict(autorange="reversed"),
        height=220,
        margin=dict(l=10, r=10, t=10, b=30)
    )
    return fig


def create_sentiment_trend_chart(df: pd.DataFrame, date_col: str, score_col: str) -> go.Figure:
    """Line chart showing average sentiment polarity over time."""
    df_sorted = df.copy()
    try:
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
        df_sorted = df_sorted.sort_values(by=date_col)
        # Group by week or date
        daily_sentiment = df_sorted.groupby(df_sorted[date_col].dt.date)[score_col].agg(['mean', 'count']).reset_index()
        daily_sentiment.columns = ['Date', 'AvgSentiment', 'ReviewCount']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_sentiment['Date'],
            y=daily_sentiment['AvgSentiment'],
            mode='lines+markers',
            line=dict(color="#6366f1", width=3, shape='spline'),
            marker=dict(size=8, color="#06b6d4", line=dict(color="#ffffff", width=1.5)),
            name="Avg Sentiment"
        ))

        fig.update_layout(
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            font=dict(color=COLOR_TEXT, family="Plus Jakarta Sans"),
            xaxis=dict(title="Review Date", gridcolor=COLOR_GRID),
            yaxis=dict(title="Avg Polarity (-1 to 1)", range=[-1.05, 1.05], gridcolor=COLOR_GRID),
            height=280,
            margin=dict(l=10, r=10, t=20, b=30)
        )
        return fig
    except Exception:
        return go.Figure()
