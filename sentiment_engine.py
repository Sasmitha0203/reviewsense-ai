"""
Sentiment Analysis & NLP Engine for Product Reviews.
Provides VADER lexicon scoring, ML TF-IDF classification,
Aspect-Based Sentiment Analysis (ABSA), Emotion cues, and text highlighting.
"""

import re
import html
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Ensure required NLTK datasets are downloaded automatically in any environment
for _resource in ['vader_lexicon', 'punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.download(_resource, quiet=True)
    except Exception:
        pass

# Initialize VADER with e-commerce specific domain additions
def create_custom_vader() -> SentimentIntensityAnalyzer:
    sia = SentimentIntensityAnalyzer()
    # Add domain-specific lexicon updates for product reviews
    product_lexicon_updates = {
        'extraordinary': 3.2,
        'worth': 2.2,
        'worthwhile': 2.4,
        'unbeatable': 2.9,
        'game-changer': 3.1,
        'game changer': 3.1,
        'perfection': 3.3,
        'phenomenal': 3.2,
        'stellar': 2.9,
        'must-buy': 2.8,
        'must buy': 2.8,
        'flawless': 3.0,
        'top-tier': 2.7,
        'top tier': 2.7,
        'audiophile': 1.9,
        'crisp': 1.8,
        'punchy': 1.8,
        'stunning': 2.8,
        'superb': 3.0,
        'durable': 2.4,
        'durability': 2.2,
        'ergonomic': 2.2,
        'overpriced': -2.6,
        'rip-off': -3.3,
        'ripoff': -3.3,
        'crackling': -2.5,
        'defective': -3.2,
        'disconnects': -2.5,
        'disconnect': -2.2,
        'useless': -3.2,
        'clamping': -1.4,
        'scratched': -2.1,
        'leak': -2.5,
        'leaks': -2.5,
        'creaking': -2.0,
        'fused': -2.0,
        'bricked': -3.6,
        'laggy': -2.4,
        'buggy': -2.5,
        'uncomfortable': -2.6,
        'headache': -2.7,
        'flimsy': -2.6,
        'plastic': -0.4,
        'plasticky': -1.8,
        'slow': -1.5,
        'crema': 1.8
    }
    sia.lexicon.update(product_lexicon_updates)
    return sia

vader_analyzer = create_custom_vader()

# Standard Product Aspects and their Keyword Mappings
PRODUCT_ASPECTS = {
    "Sound & Audio": [
        "sound", "audio", "bass", "treble", "volume", "microphone", "mic",
        "noise cancellation", "anc", "acoustic", "clarity", "vocal", "spatial",
        "loud", "speakers", "fidelity", "earcup", "earcups"
    ],
    "Battery & Power": [
        "battery", "charge", "charging", "charger", "drain", "drains",
        "power", "runtime", "endurance", "mah", "cord", "standby"
    ],
    "Comfort & Ergonomics": [
        "comfort", "comfortable", "uncomfortable", "fit", "headband",
        "cushion", "cushions", "foam", "weight", "heavy", "lightweight",
        "headache", "tight", "ears", "wrist", "strap", "padding"
    ],
    "Build Quality & Durability": [
        "build", "quality", "material", "durability", "durable", "plastic",
        "metal", "hinge", "hinges", "scratch", "sturdy", "finish", "broken",
        "crackling", "creaking", "fragile", "waterproof", "water resistance"
    ],
    "Connectivity & App": [
        "bluetooth", "connect", "connection", "disconnect", "pairing",
        "pair", "app", "sync", "latency", "lag", "software", "firmware",
        "notifications", "gps", "range", "ios", "android", "multipoint"
    ],
    "Price & Value": [
        "price", "cost", "value", "worth", "money", "expensive", "cheap",
        "affordable", "deal", "overpriced", "bargain", "investment"
    ],
    "Customer Support & Delivery": [
        "support", "service", "warranty", "customer service", "delivery",
        "shipping", "package", "packaging", "return", "refund", "seller",
        "arrival", "box"
    ]
}

EMOTION_KEYWORDS = {
    "Joy & Satisfaction": ["delighted", "love", "loved", "amazing", "great", "excellent", "happy", "perfection", "phenomenal", "thrilled", "favorite", "stellar", "pleased", "fantastic", "enjoy", "best"],
    "Admiration & Trust": ["reliable", "recommend", "recommended", "accurate", "durable", "premium", "solid", "trust", "authentic", "high-quality", "loyal", "superior", "flawless", "stunning"],
    "Frustration & Anger": ["terrible", "worst", "horrible", "awful", "scam", "waste", "garbage", "trash", "useless", "annoying", "frustrated", "frustrating", "hate", "unacceptable", "broken"],
    "Disappointment & Regret": ["disappointed", "disappointing", "regret", "expected better", "underwhelming", "poor", "mediocre", "subpar", "defective", "failed", "unfortunate", "lacks", "crackling", "leaks"],
    "Neutral & Pragmatic": ["average", "okay", "fine", "acceptable", "decent", "standard", "sufficient", "moderate", "fair"]
}

TRAINING_REVIEWS = [
    ("This product exceeded all my expectations, exceptional quality and fast shipping.", "Positive"),
    ("Best purchase I have made all year, works like a charm and very durable.", "Positive"),
    ("Outstanding battery life and crystal clear sound quality. Absolutely love it!", "Positive"),
    ("Super sleek design, feels very premium in hand. Highly recommend to everyone.", "Positive"),
    ("Worth every single penny. The active noise cancelling works like magic.", "Positive"),
    ("Five stars! Setup took 30 seconds and customer support was very helpful.", "Positive"),
    ("Fantastic performance, smooth connectivity, and comfortable for all-day wear.", "Positive"),
    ("The build quality is incredible, robust materials and immaculate attention to detail.", "Positive"),
    ("Brilliant device, runs quietly, very easy to clean and maintains steady temperature.", "Positive"),
    ("Super fast charging, great display brightness and very accurate fitness tracking.", "Positive"),
    ("Unbeatable value for the price. Performs better than models twice the cost.", "Positive"),
    ("Very pleased with this purchase. High fidelity audio and comfortable earcups.", "Positive"),
    ("Solid construction and very intuitive controls. Never had any issues.", "Positive"),
    ("Terrific customer service and the product is top notch. A++ experience!", "Positive"),
    ("Game changer for working from home. Background noise vanished completely.", "Positive"),
    ("Impressed with the battery backup and flawless Bluetooth synchronization.", "Positive"),
    ("The finish and texture feel luxurious. Beautiful unboxing experience as well.", "Positive"),
    ("Very reliable daily driver. Does everything advertised with zero hiccups.", "Positive"),

    ("It works okay, nothing particularly great or terrible. Average performance.", "Neutral"),
    ("Decent product for the price point, but build feels a bit plasticky.", "Neutral"),
    ("Sound is acceptable, though battery life is just standard at around 8 hours.", "Neutral"),
    ("It does the job as expected. Nothing more, nothing less.", "Neutral"),
    ("The hardware is good, but the companion mobile app needs some UI updates.", "Neutral"),
    ("Middle of the road quality. Fine for casual use but not for professionals.", "Neutral"),
    ("Looks nice, but the buttons are somewhat stiff to press. Mixed feelings.", "Neutral"),
    ("Fair performance for a budget item. Meets basic requirements reasonably.", "Neutral"),
    ("Shipping took a few extra days, but the product itself is satisfactory.", "Neutral"),
    ("Audio is clear enough for podcasts, but lacks deep bass for music.", "Neutral"),
    ("Usable, though manual instructions were slightly confusing to follow.", "Neutral"),
    ("Average comfort. Works well for 30 minutes before becoming a bit warm.", "Neutral"),

    ("Completely stopped working after 3 days. Total waste of money!", "Negative"),
    ("Terrible quality, broke on the first use. Customer support ignored my emails.", "Negative"),
    ("Extremely disappointed. Audio crackles constantly and Bluetooth drops out.", "Negative"),
    ("Dreadful experience. Defective battery won't hold charge for even an hour.", "Negative"),
    ("Cheap materials, flimsy plastic, and smells strange. Requesting a full refund.", "Negative"),
    ("Do not buy this product! Overpriced junk that failed within a week.", "Negative"),
    ("Very uncomfortable, gave me a massive headache within twenty minutes of wear.", "Negative"),
    ("Horrible microphone clarity. Callers couldn't understand a single word.", "Negative"),
    ("Water leaked everywhere from the bottom reservoir. Ruined my table.", "Negative"),
    ("Slow, laggy, and unresponsive touch controls. Constantly freezes.", "Negative"),
    ("Arrived damaged with scratched screen and missing accessories. Zero stars.", "Negative"),
    ("Worst customer service on the planet. Refused warranty replacement.", "Negative"),
    ("False advertising. The battery doesn't last anywhere near the claimed hours.", "Negative"),
    ("Weak build, hinges cracked right away under normal pressure. Very fragile.", "Negative")
]

class ProductSentimentClassifier:
    """TF-IDF + Logistic Regression Classifier trained for e-commerce sentiment."""
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=3000)),
            ('clf', LogisticRegression(C=2.0, max_iter=1000, class_weight='balanced'))
        ])
        self.is_trained = False
        self._train_seed_model()

    def _train_seed_model(self):
        texts, labels = zip(*TRAINING_REVIEWS)
        self.pipeline.fit(texts, labels)
        self.is_trained = True

    def train_on_data(self, texts: List[str], labels: List[str]):
        all_texts = list(texts) + [t[0] for t in TRAINING_REVIEWS]
        all_labels = list(labels) + [t[1] for t in TRAINING_REVIEWS]
        self.pipeline.fit(all_texts, all_labels)
        self.is_trained = True

    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
        predicted_label = classes[np.argmax(probs)]
        return predicted_label, prob_dict

ml_classifier = ProductSentimentClassifier()


def clean_text(text: str) -> str:
    """Sanitizes review text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def analyze_sentiment(
    text: str,
    engine: str = "Hybrid (VADER + ML)",
    pos_threshold: float = 0.05,
    neg_threshold: float = -0.05
) -> Dict[str, Any]:
    """
    Performs comprehensive sentiment analysis on a review text.
    Returns scores, classification, emotion breakdown, and aspect scores.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "label": "Neutral",
            "score": 0.0,
            "vader_scores": {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0},
            "ml_probs": {"Positive": 0.33, "Neutral": 0.34, "Negative": 0.33},
            "confidence": 0.5,
            "emotions": {},
            "aspects": {},
            "highlighted_html": ""
        }

    # 1. VADER calculation
    vader_res = vader_analyzer.polarity_scores(cleaned)
    compound = vader_res["compound"]

    if compound >= pos_threshold:
        vader_label = "Positive"
    elif compound <= neg_threshold:
        vader_label = "Negative"
    else:
        vader_label = "Neutral"

    # 2. ML Classifier calculation
    ml_label, ml_probs = ml_classifier.predict(cleaned)

    # 3. Determine final label according to selected engine
    if engine == "VADER (Lexicon)":
        final_label = vader_label
        confidence = abs(compound) if abs(compound) > 0.2 else 0.55
        score = compound
    elif engine == "Machine Learning (TF-IDF)":
        final_label = ml_label
        confidence = ml_probs.get(ml_label, 0.5)
        score = ml_probs.get("Positive", 0.0) - ml_probs.get("Negative", 0.0)
    else:
        # Hybrid Ensemble
        ml_pos = ml_probs.get("Positive", 0.0)
        ml_neg = ml_probs.get("Negative", 0.0)

        hybrid_score = (compound * 0.55) + ((ml_pos - ml_neg) * 0.45)
        score = round(float(np.clip(hybrid_score, -1.0, 1.0)), 3)

        if score >= pos_threshold:
            final_label = "Positive"
        elif score <= neg_threshold:
            final_label = "Negative"
        else:
            final_label = "Neutral"

        confidence = round(max(ml_probs.get(final_label, 0.5), (abs(compound) + 0.3) / 1.3), 3)

    # 4. Aspect-Based Sentiment Analysis
    aspects = extract_aspect_sentiment(cleaned, overall_compound=compound)

    # 5. Emotion Breakdown
    emotions = extract_emotions(cleaned, final_label)

    # 6. Word highlighting HTML
    highlighted_html = generate_highlighted_text(cleaned)

    return {
        "label": final_label,
        "score": score,
        "compound": compound,
        "vader_scores": vader_res,
        "ml_probs": ml_probs,
        "confidence": min(confidence, 0.99),
        "aspects": aspects,
        "emotions": emotions,
        "highlighted_html": highlighted_html
    }


def extract_aspect_sentiment(text: str, overall_compound: float = 0.0) -> Dict[str, Dict[str, Any]]:
    """
    Extracts product aspects mentioned in text and evaluates sentiment for each aspect.
    Uses sentence tokenizer to keep punctuation.
    """
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]

    aspect_findings: Dict[str, List[float]] = {}
    aspect_snippets: Dict[str, List[str]] = {}

    for sentence in sentences:
        s_lower = sentence.lower()
        s_score = vader_analyzer.polarity_scores(sentence)["compound"]
        
        # If sentence score is zero but contains strong words or context
        if abs(s_score) < 0.05 and abs(overall_compound) > 0.2:
            s_score = overall_compound * 0.5

        for aspect, keywords in PRODUCT_ASPECTS.items():
            for kw in keywords:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, s_lower):
                    if aspect not in aspect_findings:
                        aspect_findings[aspect] = []
                        aspect_snippets[aspect] = []
                    aspect_findings[aspect].append(s_score)
                    aspect_snippets[aspect].append(sentence)
                    break

    results = {}
    for aspect, scores in aspect_findings.items():
        avg_score = float(np.mean(scores))
        if avg_score >= 0.05:
            sentiment_tag = "Positive"
        elif avg_score <= -0.05:
            sentiment_tag = "Negative"
        else:
            sentiment_tag = "Neutral"

        results[aspect] = {
            "score": round(avg_score, 3),
            "sentiment": sentiment_tag,
            "mentions": len(scores),
            "snippets": aspect_snippets[aspect][:3]
        }

    return results


def extract_emotions(text: str, sentiment_label: str) -> Dict[str, float]:
    """Estimates emotion distribution for the product review."""
    text_lower = text.lower()
    raw_counts = {}

    for emotion, keywords in EMOTION_KEYWORDS.items():
        count = 0
        for kw in keywords:
            count += len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
        raw_counts[emotion] = count

    total = sum(raw_counts.values())
    emotions_pct = {}

    if total > 0:
        for emo, c in raw_counts.items():
            emotions_pct[emo] = round((c / total) * 100, 1)
    else:
        if sentiment_label == "Positive":
            emotions_pct = {
                "Joy & Satisfaction": 58.0,
                "Admiration & Trust": 32.0,
                "Neutral & Pragmatic": 10.0,
                "Frustration & Anger": 0.0,
                "Disappointment & Regret": 0.0
            }
        elif sentiment_label == "Negative":
            emotions_pct = {
                "Frustration & Anger": 52.0,
                "Disappointment & Regret": 38.0,
                "Neutral & Pragmatic": 10.0,
                "Joy & Satisfaction": 0.0,
                "Admiration & Trust": 0.0
            }
        else:
            emotions_pct = {
                "Neutral & Pragmatic": 65.0,
                "Admiration & Trust": 18.0,
                "Disappointment & Regret": 17.0,
                "Joy & Satisfaction": 0.0,
                "Frustration & Anger": 0.0
            }

    return emotions_pct


def generate_highlighted_text(text: str) -> str:
    """Generates HTML markup highlighting positive words in emerald green, negative in coral red."""
    words = re.findall(r'\b[\w\'-]+\b|[^\w\s]', text)
    highlighted_tokens = []

    for word in words:
        clean_w = word.lower()
        val = vader_analyzer.lexicon.get(clean_w, 0.0)
        escaped = html.escape(word)

        if val > 1.2:
            highlighted_tokens.append(
                f'<span style="background-color: rgba(16, 185, 129, 0.28); color: #10b981; font-weight: 600; padding: 2px 6px; border-radius: 4px; border-bottom: 2px solid #10b981;" title="Positive (+{val:.1f})">{escaped}</span>'
            )
        elif val < -1.2:
            highlighted_tokens.append(
                f'<span style="background-color: rgba(244, 63, 94, 0.28); color: #f43f5e; font-weight: 600; padding: 2px 6px; border-radius: 4px; border-bottom: 2px solid #f43f5e;" title="Negative ({val:.1f})">{escaped}</span>'
            )
        elif 0.5 <= val <= 1.2:
            highlighted_tokens.append(
                f'<span style="background-color: rgba(52, 211, 153, 0.18); color: #34d399; padding: 2px 4px; border-radius: 3px;" title="Mildly Positive">{escaped}</span>'
            )
        elif -1.2 <= val <= -0.5:
            highlighted_tokens.append(
                f'<span style="background-color: rgba(251, 113, 133, 0.18); color: #fb7185; padding: 2px 4px; border-radius: 3px;" title="Mildly Negative">{escaped}</span>'
            )
        else:
            highlighted_tokens.append(escaped)

    result = " ".join(highlighted_tokens)
    result = re.sub(r'\s+([,.;:!?\'])', r'\1', result)
    return result


def batch_analyze_dataframe(
    df: pd.DataFrame,
    text_col: str,
    engine: str = "Hybrid (VADER + ML)",
    pos_threshold: float = 0.05,
    neg_threshold: float = -0.05
) -> pd.DataFrame:
    """Enriches DataFrame with sentiment_label, sentiment_score, compound_score, and primary_aspect."""
    df_out = df.copy()
    labels = []
    scores = []
    compounds = []
    primary_aspects = []
    confidences = []

    for text in df_out[text_col]:
        res = analyze_sentiment(
            str(text),
            engine=engine,
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold
        )
        labels.append(res["label"])
        scores.append(res["score"])
        compounds.append(res["compound"])
        confidences.append(res["confidence"])

        if res["aspects"]:
            top_aspect = max(res["aspects"].items(), key=lambda x: x[1]["mentions"])[0]
        else:
            top_aspect = "General Experience"
        primary_aspects.append(top_aspect)

    df_out["sentiment_label"] = labels
    df_out["sentiment_score"] = scores
    df_out["vader_compound"] = compounds
    df_out["confidence"] = confidences
    df_out["primary_aspect"] = primary_aspects

    return df_out


def generate_executive_insights(df: pd.DataFrame, text_col: str) -> Dict[str, Any]:
    """Synthesizes overall metrics, key strengths, critical pain points, and seller recommendations."""
    total = len(df)
    if total == 0:
        return {}

    pos_count = (df["sentiment_label"] == "Positive").sum()
    neg_count = (df["sentiment_label"] == "Negative").sum()
    neu_count = (df["sentiment_label"] == "Neutral").sum()

    pos_pct = round((pos_count / total) * 100, 1)
    neg_pct = round((neg_count / total) * 100, 1)
    neu_pct = round((neu_count / total) * 100, 1)

    net_sentiment_score = round(pos_pct - neg_pct, 1)
    avg_score = round(float(df["sentiment_score"].mean()), 3)

    aspect_stats = {}
    for aspect in PRODUCT_ASPECTS.keys():
        aspect_stats[aspect] = {"pos": 0, "neg": 0, "total": 0}

    for text in df[text_col]:
        aspects = extract_aspect_sentiment(str(text))
        for asp, data in aspects.items():
            if asp in aspect_stats:
                aspect_stats[asp]["total"] += 1
                if data["sentiment"] == "Positive":
                    aspect_stats[asp]["pos"] += 1
                elif data["sentiment"] == "Negative":
                    aspect_stats[asp]["neg"] += 1

    strengths = []
    weaknesses = []

    for asp, stats in aspect_stats.items():
        if stats["total"] >= 2:
            pos_ratio = (stats["pos"] / stats["total"]) * 100
            neg_ratio = (stats["neg"] / stats["total"]) * 100
            if pos_ratio >= 60:
                strengths.append({
                    "aspect": asp,
                    "pos_ratio": round(pos_ratio, 1),
                    "mentions": stats["total"]
                })
            elif neg_ratio >= 30:
                weaknesses.append({
                    "aspect": asp,
                    "neg_ratio": round(neg_ratio, 1),
                    "mentions": stats["total"]
                })

    strengths = sorted(strengths, key=lambda x: x["pos_ratio"], reverse=True)
    weaknesses = sorted(weaknesses, key=lambda x: x["neg_ratio"], reverse=True)

    recommendations = []
    if any(w["aspect"] == "Battery & Power" for w in weaknesses):
        recommendations.append("🔋 **Battery Optimization**: Customers flagged battery drain or charging quirks. Conduct firmware calibration for power management and verify charger wattage specs.")
    if any(w["aspect"] == "Connectivity & App" for w in weaknesses):
        recommendations.append("📶 **Connectivity & App Stability**: Bluetooth drops or app crashes are frequent. Prioritize an immediate mobile app patch for auto-reconnect and pairing stability.")
    if any(w["aspect"] == "Comfort & Ergonomics" for w in weaknesses):
        recommendations.append("🎧 **Ergonomic Adjustment**: Noticeable reports of tight clamping force or headband pressure. Consider softer memory foam density or adjustable tension.")
    if any(w["aspect"] == "Build Quality & Durability" for w in weaknesses):
        recommendations.append("🛡️ **Quality Control & Materials**: Hinge/casing fragility was noted. Audit injection molding tolerances and reinforce high-stress joints.")
    if any(w["aspect"] == "Customer Support & Delivery" for w in weaknesses):
        recommendations.append("🤝 **Support Response SLA**: Buyers reported delayed warranty response. Implement automated ticketing confirmation and expedited replacement dispatch.")
    if not recommendations:
        if neg_pct > 25:
            recommendations.append("🔍 **Deep Dive Into Negative Outliers**: Investigate the 1-star returns to identify recurring defective batches or packaging vulnerabilities.")
        else:
            recommendations.append("🚀 **Capitalize on Strong Advocacy**: Positive reception is high (>70%). Leverage top customer praise quotes in marketing campaigns and Amazon A+ content.")

    return {
        "total_reviews": total,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "neu_pct": neu_pct,
        "net_sentiment_score": net_sentiment_score,
        "avg_score": avg_score,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
        "aspect_stats": aspect_stats
    }


def get_top_keywords(texts: List[str], top_n: int = 15) -> List[Tuple[str, int]]:
    """Extracts top non-stopword unigram/bigram keywords from review texts."""
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()

    extra_stops = {"the", "and", "a", "to", "is", "it", "of", "for", "with", "this", "in", "my", "was", "on", "i", "but", "have", "not", "so", "are"}
    stop_words.update(extra_stops)

    words_freq: Dict[str, int] = {}
    for text in texts:
        tokens = re.findall(r'\b[a-zA-Z]{3,15}\b', text.lower())
        for token in tokens:
            if token not in stop_words:
                words_freq[token] = words_freq.get(token, 0) + 1

    sorted_words = sorted(words_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]
