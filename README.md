# 🎯 ReviewSense AI — Product Review Sentiment Analysis Studio

An advanced, interactive Streamlit web application for **Product Review Sentiment Analysis** powered by **NLTK VADER**, **Scikit-learn Machine Learning (TF-IDF + Logistic Regression)**, and **Aspect-Based Sentiment Analysis (ABSA)**.

---

## ✨ Key Features

### 1. 🔍 Single Review Inspector
- **Real-Time Sentiment Scoring:** Classifies reviews as **Positive**, **Neutral**, or **Negative** with high confidence.
- **Visual Valence Highlighting:** Directly highlights words within the review (emerald green for positive attributes, coral red for pain points and defects).
- **Aspect-Based Sentiment Analysis (ABSA):** Automatically detects product dimensions (e.g. *Sound & Audio*, *Battery & Power*, *Comfort*, *Build Quality*, *Connectivity*, *Price/Value*, *Customer Support*) and rates sentiment per aspect.
- **Emotion Nuance Radar:** Measures intensity of customer emotions (*Joy & Satisfaction*, *Admiration & Trust*, *Frustration & Anger*, *Disappointment & Regret*).
- **Actionable Seller Recommendations:** Auto-generates tailored advice for product managers and customer success teams.
- **Quick-Fill Presets:** Test instant sample reviews with a single click.

### 2. 📊 Batch & Dataset Analytics
- **Preloaded Realistic Datasets:** Includes rich, multi-category review datasets (Wireless ANC Headphones, Fitness Smartwatch, Espresso Maker).
- **Custom CSV & Excel Uploader:** Upload your own product review files with auto-detection for text, star ratings, and dates.
- **Executive KPI Dashboard:** Total reviews, Positive %, Neutral %, Negative %, Net Sentiment Score (NPS), and Average Star Rating.
- **Interactive Plotly Visualizations:**
  - Sentiment Distribution Donut Chart
  - Aspect Sentiment Performance Stacked Bar Chart
  - Customer Star Rating vs Sentiment Polarity Correlation Plot
  - Temporal Sentiment Trend over time
  - Interactive Review Word Clouds (Positive vs. Negative terms)
- **Executive Insights & Friction Points:** Identifies top product strengths and critical complaints with actionable product engineering advice.
- **Filterable Explorer & CSV Export:** Search by keywords, filter by sentiment or aspect, and download the enriched dataset.

### 3. ⚔️ Product vs Competitor Benchmark
- Compare two products side-by-side on Net Sentiment Score, Positive/Negative ratio, review volume, and aspect scores.

### 4. 🧠 Model Diagnostics & Architecture
- Choose between **Hybrid Ensemble (VADER + ML)**, **VADER (Lexicon)**, or **Machine Learning (TF-IDF)**.
- Adjust sensitivity cutoff thresholds dynamically with real-time response.
- Inspect learned TF-IDF feature weights for top positive and negative vocabulary indicators.

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
miniproject/
│
├── app.py               # Main Streamlit application and layout
├── sentiment_engine.py  # NLP logic, VADER, TF-IDF Classifier, ABSA, and insights
├── ui_charts.py         # Plotly visualization components and themes
├── styles.py            # Glassmorphism dark-mode CSS and styling tokens
├── sample_data.py       # Pre-packaged multi-category review datasets
├── requirements.txt     # Python package requirements
└── README.md            # Documentation and instructions
```

---

## 🛠️ Architecture & Technologies
- **Front-End / UI:** Streamlit with custom CSS and glassmorphism styling
- **Visualizations:** Plotly Interactive Charts, Matplotlib, and WordCloud
- **NLP & Sentiment:** NLTK VADER (with custom e-commerce product domain lexicon)
- **Machine Learning:** Scikit-learn TF-IDF Vectorizer + Logistic Regression Ensemble
- **Data Manipulation:** Pandas & NumPy
