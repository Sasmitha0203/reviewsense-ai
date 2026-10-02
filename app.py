"""
Product Review Sentiment Analysis - Streamlit Web Application
Features:
- Single Review Inspector with Token Highlighting, ABSA, and Emotion Detection
- Batch & Dataset Analytics with Plotly Charts and Word Clouds
- Preloaded Datasets and Custom CSV/Excel Uploader
- Product vs Competitor Benchmark
- Multi-Engine Configuration (Hybrid, VADER, ML)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

import sentiment_engine
import sample_data
import ui_charts
from styles import CUSTOM_CSS, render_kpi

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="ReviewSense AI | Product Sentiment Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS Styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
        <span style="font-size: 2rem;">🎯</span>
        <div>
            <div style="font-family: 'Outfit'; font-size: 1.35rem; font-weight: 800; color: #ffffff;">ReviewSense AI</div>
            <div style="font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.05em; text-transform: uppercase;">Product Sentiment Studio</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Engine Settings")
    engine_choice = st.selectbox(
        "Sentiment Engine",
        ["Hybrid (VADER + ML)", "VADER (Lexicon)", "Machine Learning (TF-IDF)"],
        index=0,
        help="Select the scoring algorithm. Hybrid combines rule-based valence lexicon with statistical machine learning."
    )

    with st.expander("🎛️ Sensitivity Thresholds", expanded=False):
        pos_thresh = st.slider("Positive Cutoff (>=)", 0.00, 0.40, 0.05, 0.01)
        neg_thresh = st.slider("Negative Cutoff (<=)", -0.40, 0.00, -0.05, 0.01)

    st.markdown("---")
    st.markdown("### 📂 Quick Navigation")
    app_mode = st.radio(
        "Go to Module:",
        [
            "🔍 Single Review Inspector",
            "📊 Batch & Dataset Analytics",
            "⚔️ Product Benchmark",
            "🧠 Model Diagnostics & Architecture"
        ]
    )

    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px; font-size: 0.8rem; color: #10b981;">
        ● <b>Engine Active:</b> Ready for real-time inference
    </div>
    """, unsafe_allow_html=True)


# Main Header
st.markdown("""
<div class="hero-title">
    <span>🎯 Product Review Sentiment Analysis</span>
</div>
<div class="hero-subtitle">
    Harness natural language processing, Aspect-Based Sentiment Analysis (ABSA), and machine learning to decode customer feedback.
</div>
""", unsafe_allow_html=True)


# MODULE 1: SINGLE REVIEW INSPECTOR
if app_mode == "🔍 Single Review Inspector":
    st.markdown("### 🔍 Real-Time Review Inspector")
    st.write("Analyze individual product reviews to extract polarity, emotional nuances, and aspect-specific opinions.")

    # Preset Sample Reviews for Instant Click-to-Test
    st.markdown("<span style='font-size: 0.85rem; color: #94a3b8; font-weight: 600;'>TRY A SAMPLE REVIEW:</span>", unsafe_allow_html=True)
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)

    preset_review = ""
    if sample_col1.button("🌟 5-Star Rave (Headphones)", use_container_width=True):
        preset_review = "The active noise cancellation is world-class! Battery lasts easily over 32 hours and the bass is punchy without distorting vocals. Highly recommend for frequent travelers!"
    if sample_col2.button("❌ 1-Star Defect (Hardware)", use_container_width=True):
        preset_review = "Terrible quality control. The left speaker started crackling and died after 14 days of light use. Customer service has ignored my warranty ticket for 5 days. Total rip-off!"
    if sample_col3.button("⚖️ 3-Star Mixed (Good sound, tight fit)", use_container_width=True):
        preset_review = "Audio quality is very rich and pristine, but the clamping force is rather tight for larger heads. Caused a headache after 45 minutes of wear. Decent battery though."
    if sample_col4.button("⌚ Smartwatch (Battery & Screen)", use_container_width=True):
        preset_review = "AMOLED display is vivid and readable under direct sunlight. Battery lasts 6 full days easily. However, step count is slightly inaccurate when driving."

    review_input = st.text_area(
        "Enter or paste customer review text:",
        value=preset_review if preset_review else "The active noise cancellation is world-class! Battery lasts easily over 32 hours and the bass is punchy without distorting vocals. Highly recommend for frequent travelers!",
        height=130,
        placeholder="Type customer review here..."
    )

    analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=False)

    if review_input.strip():
        with st.spinner("Analyzing review with NLP engine..."):
            res = sentiment_engine.analyze_sentiment(
                review_input,
                engine=engine_choice,
                pos_threshold=pos_thresh,
                neg_threshold=neg_thresh
            )

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        # Top Metric Row
        label = res["label"]
        score = res["score"]
        conf = res["confidence"]
        badge_class = "badge-pos" if label == "Positive" else ("badge-neg" if label == "Negative" else "badge-neu")
        badge_icon = "🟢" if label == "Positive" else ("🔴" if label == "Negative" else "🟡")

        m1, m2, m3, m4 = st.columns([1.2, 1, 1, 1.8])
        with m1:
            st.markdown(f"""
            <div class="kpi-container" style="height: 100%;">
                <div class="kpi-label">Sentiment Classification</div>
                <div style="margin-top: 8px;">
                    <span class="{badge_class}" style="font-size: 1.15rem; padding: 6px 16px;">
                        {badge_icon} {label}
                    </span>
                </div>
                <div class="kpi-sub" style="margin-top: 10px;">Scored via {engine_choice}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(render_kpi("Model Confidence", f"{conf * 100:.1f}%", "Ensemble certainty", "#6366f1"), unsafe_allow_html=True)

        with m3:
            st.markdown(render_kpi("Polarity Score", f"{score:+.2f}", "Range: -1.00 to +1.00", "#10b981" if score > 0 else "#f43f5e"), unsafe_allow_html=True)

        with m4:
            gauge_fig = ui_charts.create_gauge_chart(score, title="Sentiment Polarity Meter")
            st.plotly_chart(gauge_fig, use_container_width=True)

        # Detailed Analysis Tabs
        tab_highlight, tab_aspects, tab_emotions, tab_recommendation = st.tabs([
            "✨ Token Valence Highlight",
            "🏷️ Aspect-Based Sentiment (ABSA)",
            "🎭 Emotional Nuances",
            "💡 Seller Recommendation"
        ])

        with tab_highlight:
            st.markdown("#### Visual Sentiment Breakdown")
            st.write("Words contributing to sentiment are highlighted: <span style='color: #10b981; font-weight: bold;'>Green</span> for positive words, <span style='color: #f43f5e; font-weight: bold;'>Red</span> for negative complaints.", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="highlight-box">
                {res["highlighted_html"]}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Raw Model Breakdown", expanded=False):
                c_a, c_b = st.columns(2)
                with c_a:
                    st.write("**VADER Scores:**")
                    st.json(res["vader_scores"])
                with c_b:
                    st.write("**Machine Learning Class Probabilities:**")
                    st.json(res["ml_probs"])

        with tab_aspects:
            st.markdown("#### Detected Product Aspects")
            st.write("Aspect-Based Sentiment Analysis pinpoints how the reviewer feels about specific product features:")

            if res["aspects"]:
                aspect_cols = st.columns(min(len(res["aspects"]), 3))
                for idx, (aspect_name, asp_data) in enumerate(res["aspects"].items()):
                    col_target = aspect_cols[idx % 3]
                    with col_target:
                        asp_sentiment = asp_data["sentiment"]
                        asp_color = "#10b981" if asp_sentiment == "Positive" else ("#f43f5e" if asp_sentiment == "Negative" else "#f59e0b")
                        asp_badge = "badge-pos" if asp_sentiment == "Positive" else ("badge-neg" if asp_sentiment == "Negative" else "badge-neu")

                        snippets_html = "".join([f"<li style='color: #cbd5e1; font-size: 0.85rem; margin-bottom: 4px;'>“{snip}”</li>" for snip in asp_data["snippets"]])

                        st.markdown(f"""
                        <div class="aspect-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <b style="color: #ffffff; font-size: 0.95rem;">{aspect_name}</b>
                                <span class="{asp_badge}">{asp_sentiment}</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 6px;">Mentions: {asp_data['mentions']} | Score: {asp_data['score']:+.2f}</div>
                            <ul style="padding-left: 18px; margin: 0;">
                                {snippets_html}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No specific hardware aspect keywords were detected in this review. The sentiment reflects the reviewer's overall experience.")

        with tab_emotions:
            st.markdown("#### Customer Emotion Intensity")
            st.write("Estimated distribution of emotional signals inferred from text cues and vocabulary:")
            emo_chart = ui_charts.create_emotion_bar_chart(res["emotions"])
            st.plotly_chart(emo_chart, use_container_width=True)

        with tab_recommendation:
            st.markdown("#### 💡 Actionable Customer Experience Takeaway")
            if label == "Positive":
                st.markdown("""
                <div class="recommendation-box">
                    <b style="color: #10b981; font-size: 1.05rem;">🌟 Advocate Opportunity</b><br>
                    <p style="color: #cbd5e1; margin-top: 6px;">
                        This customer is highly satisfied. Consider inviting them to join your VIP review program or user community. Highlight specific praise points (such as audio clarity or battery endurance) in social proof marketing materials.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif label == "Negative":
                st.markdown("""
                <div class="recommendation-box" style="border-left-color: #f43f5e; background: linear-gradient(135deg, rgba(244, 63, 94, 0.1) 0%, rgba(244, 63, 94, 0.04) 100%);">
                    <b style="color: #f43f5e; font-size: 1.05rem;">🚨 High Priority Escalation</b><br>
                    <p style="color: #cbd5e1; margin-top: 6px;">
                        Immediate retention action advised. The reviewer reported significant frustration with hardware defects or support response. Trigger a high-priority customer support ticket offering an expedited warranty replacement or return authorization.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="recommendation-box" style="border-left-color: #f59e0b; background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.04) 100%);">
                    <b style="color: #f59e0b; font-size: 1.05rem;">⚖️ Constructive Product Feedback</b><br>
                    <p style="color: #cbd5e1; margin-top: 6px;">
                        This review contains mixed impressions. The core product functionality is recognized, but specific usability frictions (e.g. software, clamping force, or documentation) prevented a 5-star experience. Feed these insights into the next product revision cycle.
                    </p>
                </div>
                """, unsafe_allow_html=True)


# MODULE 2: BATCH & DATASET ANALYTICS
elif app_mode == "📊 Batch & Dataset Analytics":
    st.markdown("### 📊 Batch Review Dataset Analytics")
    st.write("Analyze entire product review datasets to uncover high-level sentiment trends, aspect strengths, and friction points.")

    # Data Source Options
    source_type = st.radio(
        "Choose Data Source:",
        ["📦 Use Preloaded Realistic Dataset", "📤 Upload Custom File (CSV or Excel)"],
        horizontal=True
    )

    df_raw = None
    product_display_name = ""

    if source_type == "📦 Use Preloaded Realistic Dataset":
        samples = sample_data.get_sample_datasets()
        chosen_sample = st.selectbox("Select Sample Product Dataset:", list(samples.keys()))
        df_raw = samples[chosen_sample]
        product_display_name = chosen_sample
        st.caption(f"Loaded **{len(df_raw)}** reviews for *{chosen_sample}*.")

    else:
        uploaded_file = st.file_uploader("Upload CSV or Excel file of customer reviews:", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    df_raw = pd.read_excel(uploaded_file)
                product_display_name = uploaded_file.name
                st.success(f"Successfully uploaded {uploaded_file.name} ({len(df_raw)} rows)")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    if df_raw is not None and not df_raw.empty:
        # Detect or select columns
        cols = list(df_raw.columns)
        
        # Smart column auto-detection
        default_text_col = next((c for c in cols if any(k in c.lower() for k in ["text", "review", "comment", "body", "content"])), cols[0])
        default_rating_col = next((c for c in cols if any(k in c.lower() for k in ["rating", "score", "star", "stars"])), None)
        default_date_col = next((c for c in cols if any(k in c.lower() for k in ["date", "time", "created"])), None)

        with st.expander("⚙️ Dataset Column Mapping", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                text_col = st.selectbox("Review Text Column:", cols, index=cols.index(default_text_col))
            with c2:
                rating_options = ["None"] + cols
                rating_idx = (cols.index(default_rating_col) + 1) if default_rating_col else 0
                rating_col = st.selectbox("Rating Column (Optional):", rating_options, index=rating_idx)
                if rating_col == "None":
                    rating_col = None
            with c3:
                date_options = ["None"] + cols
                date_idx = (cols.index(default_date_col) + 1) if default_date_col else 0
                date_col = st.selectbox("Date Column (Optional):", date_options, index=date_idx)
                if date_col == "None":
                    date_col = None

        # Execute Batch Analysis
        with st.spinner("Processing batch sentiment analysis and aspect tagging..."):
            df_analyzed = sentiment_engine.batch_analyze_dataframe(
                df_raw,
                text_col=text_col,
                engine=engine_choice,
                pos_threshold=pos_thresh,
                neg_threshold=neg_thresh
            )
            insights = sentiment_engine.generate_executive_insights(df_analyzed, text_col=text_col)

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        # KPI Dashboard Cards
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(render_kpi("Total Reviews", f"{insights['total_reviews']:,}", "Analyzed sample", "#6366f1"), unsafe_allow_html=True)
        with k2:
            st.markdown(render_kpi("Positive %", f"{insights['pos_pct']}%", f"{(df_analyzed['sentiment_label'] == 'Positive').sum()} reviews", "#10b981"), unsafe_allow_html=True)
        with k3:
            st.markdown(render_kpi("Neutral %", f"{insights['neu_pct']}%", f"{(df_analyzed['sentiment_label'] == 'Neutral').sum()} reviews", "#f59e0b"), unsafe_allow_html=True)
        with k4:
            st.markdown(render_kpi("Negative %", f"{insights['neg_pct']}%", f"{(df_analyzed['sentiment_label'] == 'Negative').sum()} reviews", "#f43f5e"), unsafe_allow_html=True)
        with k5:
            nps_val = insights['net_sentiment_score']
            nps_color = "#10b981" if nps_val > 20 else ("#f43f5e" if nps_val < -10 else "#f59e0b")
            st.markdown(render_kpi("Net Sentiment", f"{nps_val:+0.1f}", "Pos % minus Neg %", nps_color), unsafe_allow_html=True)
        with k6:
            if rating_col and rating_col in df_analyzed.columns:
                avg_star = df_analyzed[rating_col].astype(float).mean()
                st.markdown(render_kpi("Avg Star Rating", f"{avg_star:.2f} ★", "Customer rating", "#f59e0b"), unsafe_allow_html=True)
            else:
                st.markdown(render_kpi("Avg Polarity", f"{insights['avg_score']:+.2f}", "Polarity index", "#06b6d4"), unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        # Charts Row 1: Donut Distribution & Aspect Breakdown
        row1_col1, row1_col2 = st.columns([1, 1.6])

        with row1_col1:
            st.markdown("#### 🍩 Sentiment Distribution")
            sentiment_counts = {
                "Positive": int((df_analyzed["sentiment_label"] == "Positive").sum()),
                "Neutral": int((df_analyzed["sentiment_label"] == "Neutral").sum()),
                "Negative": int((df_analyzed["sentiment_label"] == "Negative").sum())
            }
            donut_fig = ui_charts.create_sentiment_donut(sentiment_counts)
            st.plotly_chart(donut_fig, use_container_width=True)

        with row1_col2:
            st.markdown("#### 🏷️ Aspect Sentiment Breakdown")
            aspect_fig = ui_charts.create_aspect_breakdown_chart(insights["aspect_stats"])
            if aspect_fig.data:
                st.plotly_chart(aspect_fig, use_container_width=True)
            else:
                st.info("No aspect matches detected across dataset.")

        # Charts Row 2: Star Rating Correlation & Word Cloud
        row2_col1, row2_col2 = st.columns([1.2, 1.2])

        with row2_col1:
            if rating_col and rating_col in df_analyzed.columns:
                st.markdown("#### ⭐ Star Rating vs Sentiment Polarity")
                scatter_fig = ui_charts.create_rating_sentiment_scatter(df_analyzed, rating_col, "sentiment_score")
                st.plotly_chart(scatter_fig, use_container_width=True)
            elif date_col and date_col in df_analyzed.columns:
                st.markdown("#### 📈 Sentiment Trend Over Time")
                trend_fig = ui_charts.create_sentiment_trend_chart(df_analyzed, date_col, "sentiment_score")
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.markdown("#### 🔑 Top Positive & Negative Keywords")
                pos_texts = df_analyzed[df_analyzed["sentiment_label"] == "Positive"][text_col].tolist()
                top_pos = sentiment_engine.get_top_keywords(pos_texts, top_n=6)
                neg_texts = df_analyzed[df_analyzed["sentiment_label"] == "Negative"][text_col].tolist()
                top_neg = sentiment_engine.get_top_keywords(neg_texts, top_n=6)

                c_p, c_n = st.columns(2)
                with c_p:
                    st.write("**Top Positive Terms:**")
                    for w, count in top_pos:
                        st.markdown(f"<span class='badge-pos'>{w} ({count})</span>", unsafe_allow_html=True)
                with c_n:
                    st.write("**Top Negative Terms:**")
                    for w, count in top_neg:
                        st.markdown(f"<span class='badge-neg'>{w} ({count})</span>", unsafe_allow_html=True)

        with row2_col2:
            st.markdown("#### ☁️ Review Word Cloud")
            cloud_sentiment = st.radio("Generate Word Cloud for:", ["All Reviews", "Positive Only", "Negative Only"], horizontal=True)
            
            if cloud_sentiment == "Positive Only":
                target_texts = df_analyzed[df_analyzed["sentiment_label"] == "Positive"][text_col]
                colormap = "Greens"
            elif cloud_sentiment == "Negative Only":
                target_texts = df_analyzed[df_analyzed["sentiment_label"] == "Negative"][text_col]
                colormap = "Reds"
            else:
                target_texts = df_analyzed[text_col]
                colormap = "cool"

            combined_text = " ".join([str(t) for t in target_texts if str(t).strip()])
            if combined_text:
                try:
                    wc = WordCloud(
                        width=600,
                        height=320,
                        background_color="#0b0f19",
                        colormap=colormap,
                        max_words=80,
                        stopwords={"the", "and", "to", "a", "is", "it", "of", "for", "with", "this", "my", "was", "but", "in", "on", "have"}
                    ).generate(combined_text)

                    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0b0f19")
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    plt.tight_layout(pad=0)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.caption(f"WordCloud rendering skipped: {e}")
            else:
                st.caption("Not enough text data to render word cloud.")

        # Executive Insights & Actionable Takeaways
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 📋 Executive Summary & Recommendations")

        col_str, col_weak = st.columns(2)
        with col_str:
            st.markdown("##### 🏆 Core Strengths (Customer Praises)")
            if insights["strengths"]:
                for s in insights["strengths"]:
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                        <b style="color: #10b981;">{s['aspect']}</b> — {s['pos_ratio']}% Positive ({s['mentions']} mentions)
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No standout strengths (>60% positive) identified in this dataset.")

        with col_weak:
            st.markdown("##### ⚠️ Pain Points (Customer Friction)")
            if insights["weaknesses"]:
                for w in insights["weaknesses"]:
                    st.markdown(f"""
                    <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                        <b style="color: #f43f5e;">{w['aspect']}</b> — {w['neg_ratio']}% Negative ({w['mentions']} mentions)
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No dominant pain points (>30% negative) flagged.")

        st.markdown("##### 💡 Strategic Recommendations for Product & Seller Teams")
        for rec in insights["recommendations"]:
            st.markdown(f"""
            <div class="recommendation-box" style="margin-bottom: 8px;">
                <span style="color: #e2e8f0; font-size: 0.95rem;">{rec}</span>
            </div>
            """, unsafe_allow_html=True)

        # Interactive Review Explorer Table & Export
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 🔍 Filterable Review Explorer")

        f_col1, f_col2, f_col3 = st.columns([1, 1, 1.5])
        with f_col1:
            filter_sentiment = st.multiselect("Filter by Sentiment:", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
        with f_col2:
            unique_aspects = ["All"] + sorted(list(df_analyzed["primary_aspect"].unique()))
            filter_aspect = st.selectbox("Filter by Primary Aspect:", unique_aspects)
        with f_col3:
            search_query = st.text_input("Search keywords:", placeholder="Search reviews...")

        df_filtered = df_analyzed.copy()
        if filter_sentiment:
            df_filtered = df_filtered[df_filtered["sentiment_label"].isin(filter_sentiment)]
        if filter_aspect != "All":
            df_filtered = df_filtered[df_filtered["primary_aspect"] == filter_aspect]
        if search_query.strip():
            df_filtered = df_filtered[df_filtered[text_col].astype(str).str.contains(search_query, case=False, na=False)]

        st.dataframe(
            df_filtered,
            use_container_width=True,
            column_config={
                "sentiment_label": st.column_config.TextColumn("Sentiment"),
                "sentiment_score": st.column_config.ProgressColumn("Polarity", min_value=-1.0, max_value=1.0, format="%.2f"),
                "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            },
            height=300
        )

        # Export CSV Button
        csv_data = df_analyzed.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Enriched Dataset (CSV)",
            data=csv_data,
            file_name=f"enriched_sentiment_{product_display_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            help="Download the complete dataset with sentiment labels, polarity scores, and aspect tags."
        )


# MODULE 3: PRODUCT BENCHMARK
elif app_mode == "⚔️ Product Benchmark":
    st.markdown("### ⚔️ Product vs Competitor Benchmark")
    st.write("Compare customer sentiment metrics side-by-side to understand relative competitive advantages.")

    samples = sample_data.get_sample_datasets()
    product_keys = list(samples.keys())

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        prod_a_name = st.selectbox("Select Product A:", product_keys, index=0)
        df_a = samples[prod_a_name]
    with b_col2:
        prod_b_name = st.selectbox("Select Product B:", product_keys, index=1)
        df_b = samples[prod_b_name]

    if prod_a_name == prod_b_name:
        st.warning("Please choose two different products to run a meaningful comparison.")
    else:
        df_a_scored = sentiment_engine.batch_analyze_dataframe(df_a, text_col="text", engine=engine_choice)
        df_b_scored = sentiment_engine.batch_analyze_dataframe(df_b, text_col="text", engine=engine_choice)

        ins_a = sentiment_engine.generate_executive_insights(df_a_scored, text_col="text")
        ins_b = sentiment_engine.generate_executive_insights(df_b_scored, text_col="text")

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        comp1, comp2 = st.columns(2)
        with comp1:
            st.markdown(f"#### 🎧 {prod_a_name}")
            c_k1, c_k2, c_k3 = st.columns(3)
            with c_k1:
                st.markdown(render_kpi("Positive %", f"{ins_a['pos_pct']}%", f"{ins_a['total_reviews']} reviews", "#10b981"), unsafe_allow_html=True)
            with c_k2:
                st.markdown(render_kpi("Negative %", f"{ins_a['neg_pct']}%", "Complaints", "#f43f5e"), unsafe_allow_html=True)
            with c_k3:
                st.markdown(render_kpi("Net Sentiment", f"{ins_a['net_sentiment_score']:+0.1f}", "NPS Index", "#6366f1"), unsafe_allow_html=True)

            st.plotly_chart(ui_charts.create_aspect_breakdown_chart(ins_a["aspect_stats"]), use_container_width=True)

        with comp2:
            st.markdown(f"#### ⌚ {prod_b_name}")
            c_k4, c_k5, c_k6 = st.columns(3)
            with c_k4:
                st.markdown(render_kpi("Positive %", f"{ins_b['pos_pct']}%", f"{ins_b['total_reviews']} reviews", "#10b981"), unsafe_allow_html=True)
            with c_k5:
                st.markdown(render_kpi("Negative %", f"{ins_b['neg_pct']}%", "Complaints", "#f43f5e"), unsafe_allow_html=True)
            with c_k6:
                st.markdown(render_kpi("Net Sentiment", f"{ins_b['net_sentiment_score']:+0.1f}", "NPS Index", "#6366f1"), unsafe_allow_html=True)

            st.plotly_chart(ui_charts.create_aspect_breakdown_chart(ins_b["aspect_stats"]), use_container_width=True)


# MODULE 4: MODEL DIAGNOSTICS & ARCHITECTURE
elif app_mode == "🧠 Model Diagnostics & Architecture":
    st.markdown("### 🧠 Model Diagnostics & Architecture")
    st.write("Inspect how the sentiment models are built, their feature weights, and underlying NLP architecture.")

    arch_col1, arch_col2 = st.columns(2)

    with arch_col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #6366f1; margin-top: 0;">1. Valence Aware Dictionary (VADER)</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                A specialized rule-based sentiment model tuned for social and e-commerce contexts. 
                It evaluates valence with punctuation weight (e.g. <i>"good!"</i> vs <i>"good!!!"</i>), 
                capitalization boosts (e.g. <i>"AMAZING"</i>), negations (e.g. <i>"not bad"</i>), and intensifiers.
            </p>
            <ul style="color: #94a3b8; font-size: 0.85rem;">
                <li>Speed: &lt; 0.5 milliseconds per review</li>
                <li>Customized with e-commerce lexicon additions (flawless, defective, crackling, etc.)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #06b6d4; margin-top: 0;">2. Machine Learning TF-IDF Classifier</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                A Scikit-learn Pipeline with unigram & bigram TF-IDF vectorization paired with L2-regularized 
                Logistic Regression. Trained on curated e-commerce review patterns.
            </p>
            <ul style="color: #94a3b8; font-size: 0.85rem;">
                <li>N-gram range: (1, 2)</li>
                <li>Provides calibrated class probabilities: P(Pos), P(Neu), P(Neg)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with arch_col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #10b981; margin-top: 0;">3. Aspect-Based Sentiment Analysis (ABSA)</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                Segments user reviews into sentence-level clauses and maps semantic keywords to 7 primary product dimensions:
            </p>
            <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                <span class="badge-neu">Sound & Audio</span>
                <span class="badge-neu">Battery & Power</span>
                <span class="badge-neu">Comfort & Ergonomics</span>
                <span class="badge-neu">Build Quality & Durability</span>
                <span class="badge-neu">Connectivity & App</span>
                <span class="badge-neu">Price & Value</span>
                <span class="badge-neu">Customer Support & Delivery</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #f59e0b; margin-top: 0;">4. Hybrid Ensemble Fusion</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                Combines lexicon compound valence with ML predictive probability:
                <code>Score = (VADER * 0.55) + ((P_pos - P_neg) * 0.45)</code>.
                Achieves superior handling of edge cases and nuanced sentiment.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 🔬 Inspect Machine Learning Feature Weights")
    clf = sentiment_engine.ml_classifier.pipeline.named_steps['clf']
    tfidf = sentiment_engine.ml_classifier.pipeline.named_steps['tfidf']
    feature_names = np.array(tfidf.get_feature_names_out())

    # Get positive and negative coefficient top features
    classes = list(clf.classes_)
    if "Positive" in classes and "Negative" in classes:
        pos_idx = classes.index("Positive")
        neg_idx = classes.index("Negative")

        pos_coefs = clf.coef_[pos_idx]
        top_pos_indices = np.argsort(pos_coefs)[-12:][::-1]
        top_pos_features = feature_names[top_pos_indices]
        top_pos_weights = pos_coefs[top_pos_indices]

        neg_coefs = clf.coef_[neg_idx]
        top_neg_indices = np.argsort(neg_coefs)[-12:][::-1]
        top_neg_features = feature_names[top_neg_indices]
        top_neg_weights = neg_coefs[top_neg_indices]

        fw_col1, fw_col2 = st.columns(2)
        with fw_col1:
            st.write("**Top Learned Positive Indicators:**")
            df_pos_feats = pd.DataFrame({"Term": top_pos_features, "Weight": top_pos_weights})
            st.dataframe(df_pos_feats, use_container_width=True, height=220)
        with fw_col2:
            st.write("**Top Learned Negative Indicators:**")
            df_neg_feats = pd.DataFrame({"Term": top_neg_features, "Weight": top_neg_weights})
            st.dataframe(df_neg_feats, use_container_width=True, height=220)


# Footer
st.markdown("""
<div class="custom-footer">
    Product Review Sentiment Analysis Studio | Built with Streamlit, NLTK VADER, Scikit-learn, and Plotly
</div>
""", unsafe_allow_html=True)
