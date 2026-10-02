"""
Premium Dark-Mode Glassmorphism Design System and Custom CSS for Streamlit.
"""

CUSTOM_CSS = """
<style>
/* Import Modern Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0b0f19;
    --bg-card: rgba(18, 24, 38, 0.7);
    --bg-card-hover: rgba(28, 38, 58, 0.8);
    --border-color: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(99, 102, 241, 0.3);
    
    --accent-indigo: #6366f1;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header & Titles */
h1, h2, h3, h4, .stSubheader {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Main Container Header Gradient */
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 20%, #94a3b8 50%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 1.5rem;
}

/* Glassmorphism Card Container */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
    transition: all 0.25s ease-in-out;
    margin-bottom: 1rem;
}

.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45);
}

/* KPI Metric Cards */
.kpi-container {
    display: flex;
    flex-direction: column;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 16px 18px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.kpi-container:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
}

.kpi-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.kpi-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
}

.kpi-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 4px;
}

/* Sentiment Badges */
.badge-pos {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 700;
}

.badge-neg {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(244, 63, 94, 0.15);
    color: #f43f5e;
    border: 1px solid rgba(244, 63, 94, 0.3);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 700;
}

.badge-neu {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 700;
}

/* Aspect Pill Tags */
.aspect-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

/* Recommendation Box */
.recommendation-box {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.06) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
}

/* Tabs Styling */
button[data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 10px 20px !important;
}

/* Highlighted Text Card */
.highlight-box {
    background: rgba(10, 15, 29, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 18px 20px;
    line-height: 1.8;
    font-size: 1.02rem;
    color: #e2e8f0;
}

/* Footer styling */
.custom-footer {
    text-align: center;
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}
</style>
"""

def render_kpi(label: str, value: str, sub: str = "", delta_color: str = "#6366f1") -> str:
    """Renders a sleek HTML KPI card."""
    return f"""
    <div class="kpi-container">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {delta_color};">{value}</div>
        {f'<div class="kpi-sub">{sub}</div>' if sub else ''}
    </div>
    """
