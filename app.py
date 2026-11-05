"""
Customer Feedback Analyzer — Streamlit dashboard
Run:  streamlit run app.py

Uses the fine-tuned DistilBERT from the notebook (models/distilbert-feedback) if present,
otherwise falls back to the free pretrained RoBERTa sentiment model.
"""
import os
import pandas as pd
import streamlit as st
from transformers import pipeline

LOCAL_MODEL = "models/distilbert-feedback"
FALLBACK_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_CATEGORIES = "delivery, customer service, refund and returns, account and payment, prime membership, product quality, price"
COLORS = {"negative": "#e74c3c", "neutral": "#f1c40f", "positive": "#2ecc71"}

st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="💬", layout="wide")


# ---------------- Model loading (cached) ----------------
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment():
    path = LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else FALLBACK_MODEL
    return pipeline("sentiment-analysis", model=path, truncation=True, max_length=512), path


@st.cache_resource(show_spinner="Loading category model...")
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


@st.cache_resource(show_spinner="Loading aspect model...")
def load_absa():
    return pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")


def predict_sentiment(texts):
    clf, _ = load_sentiment()
    # 256 tokens = the length the model was fine-tuned on; also twice as fast as 512
    out = clf(list(texts), batch_size=16, truncation=True, max_length=256)
    return [o["label"].lower() for o in out], [round(o["score"], 3) for o in out]


def run_with_progress(label, fn, items, chunk, extra=()):
    """Run fn over items in chunks with a progress bar; results are kept for the session,
    so changing a chart option does not re-run the model."""
    store = st.session_state.setdefault("bulk_results", {})
    key = (label, hash(tuple(items)), tuple(extra))
    if key in store:
        return store[key]
    bar = st.progress(0.0, text=f"{label}...")
    out = []
    for i in range(0, len(items), chunk):
        out.extend(fn(items[i:i + chunk]))
        done = min(i + chunk, len(items))
        bar.progress(done / len(items), text=f"{label}: {done} / {len(items)} reviews")
    bar.empty()
    store[key] = out
    return out


def sentiment_chunk(texts):
    labels, scores = predict_sentiment(texts)
    return list(zip(labels, scores))


def category_chunk(texts, cats):
    out = load_zero_shot()(list(texts), candidate_labels=list(cats), batch_size=4)
    out = out if isinstance(out, list) else [out]
    return [o["labels"][0] for o in out]


def aspect_sentiment(text, aspects):
    absa = load_absa()
    rows = []
    for a in aspects:
        r = absa({"text": text[:1500], "text_pair": a})
        r = r[0] if isinstance(r, list) else r
        rows.append({"Aspect": a, "Sentiment": r["label"].lower(), "Confidence": round(r["score"], 3)})
    return pd.DataFrame(rows)


# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Settings")
categories = [c.strip() for c in st.sidebar.text_area(
    "Feedback categories (comma-separated)", DEFAULT_CATEGORIES).split(",") if c.strip()]
use_categories = st.sidebar.checkbox("Detect categories in single review", value=True,
                                     help="On a laptop CPU, turn this off for bulk files with more than ~100 rows")
use_absa = st.sidebar.checkbox("Aspect-based sentiment (slower)", value=True)
_, model_used = load_sentiment()
st.sidebar.caption(f"Sentiment model: `{model_used}`")

st.title("💬 Customer Feedback Analyzer")
st.caption("NLP project · Sentiment, categories and aspect-level opinions using free open-source models")

tab1, tab2 = st.tabs(["🔍 Single feedback", "📊 Bulk analysis (CSV)"])

# ---------------- Tab 1: single review ----------------
with tab1:
    text = st.text_area("Paste a customer review",
                        "The product quality is great but delivery was a week late and customer service kept me on hold for an hour.",
                        height=120)
    if st.button("Analyze", type="primary") and text.strip():
        labels, scores = predict_sentiment([text])
        c1, c2 = st.columns(2)
        c1.metric("Overall sentiment", labels[0].capitalize())
        c2.metric("Confidence", f"{scores[0]:.0%}")

        if use_categories and categories:
            zs = load_zero_shot()(text, candidate_labels=categories, multi_label=True)
            cat = pd.DataFrame({"Category": zs["labels"], "Relevance": zs["scores"]})
            st.subheader("Categories mentioned")
            st.bar_chart(cat.set_index("Category"))
            relevant = cat[cat.Relevance > 0.5]["Category"].tolist() or [zs["labels"][0]]
        else:
            relevant = categories

        if use_absa and relevant:
            st.subheader("Aspect-level sentiment")
            st.dataframe(aspect_sentiment(text, relevant), width="stretch", hide_index=True)

# ---------------- Tab 2: bulk CSV ----------------
with tab2:
    st.write("Upload **Amazon_Reviews.csv** (or any CSV with a review text column; a date column enables trends).")
    file = st.file_uploader("CSV file", type="csv")
    if file:
        df = pd.read_csv(file, engine="python", on_bad_lines="skip")
        cols = list(df.columns)
        guess_text = next((c for c in ["Review Text", "text", "review", "Review"] if c in cols), cols[0])
        guess_date = next((c for c in ["Review Date", "date", "Date"] if c in cols), None)
        c1, c2 = st.columns(2)
        text_col = c1.selectbox("Review text column", cols, index=cols.index(guess_text))
        date_opts = ["(none)"] + cols
        date_col = c2.selectbox("Date column", date_opts,
                                index=date_opts.index(guess_date) if guess_date else 0)

        df = df.dropna(subset=[text_col]).reset_index(drop=True)
        df["text"] = df[text_col].astype(str)
        if date_col != "(none)":
            df["date"] = df[date_col]
        if "Rating" in df.columns:
            df["stars"] = pd.to_numeric(df["Rating"].astype(str).str.extract(r"(\d)")[0], errors="coerce")
        # Warn when the chosen columns don't look like review text / dates
        avg_words = df["text"].str.split().str.len().mean()
        if avg_words < 10:   # real reviews average ~80 words; titles ~5, countries/dates 1-3
            st.warning(f"The column '{text_col}' has only {avg_words:.1f} words per row on average - "
                       "it does not look like review text. For Amazon_Reviews.csv choose 'Review Text'.")
        if "date" in df.columns:
            valid_dates = pd.to_datetime(df["date"], errors="coerce", utc=True).notna().mean()
            if valid_dates < 0.5:
                st.warning(f"The column '{date_col}' does not contain dates, so the trend chart is skipped. "
                           "For Amazon_Reviews.csv choose 'Review Date'.")
                df = df.drop(columns="date")

        if len(df) > 10:
            c3, c4 = st.columns([3, 1])
            max_rows = c3.slider("Rows to analyze", 10, min(len(df), 2000), min(len(df), 300))
            random_sample = c4.checkbox("Random sample", value=True,
                                        help="Picks rows from across the whole file, so trends cover several years")
            df = (df.sample(max_rows, random_state=42) if random_sample else df.head(max_rows)).reset_index(drop=True)

        bulk_cats = st.checkbox("Also detect categories (slow on a laptop CPU - use 50 rows or fewer)",
                                value=False)

        res = run_with_progress("Analyzing sentiment", sentiment_chunk, list(df["text"].str[:2000]), 16)
        df["sentiment"] = [r[0] for r in res]
        df["confidence"] = [r[1] for r in res]

        if bulk_cats and categories:
            # the first ~400 characters usually name the subject of the review; much faster than the full text
            df["category"] = run_with_progress("Detecting categories",
                                               lambda t: category_chunk(t, categories),
                                               list(df["text"].str[:400]), 4, extra=categories)

        counts = df["sentiment"].value_counts()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Reviews", len(df))
        for col, lab in zip((k2, k3, k4), ("positive", "neutral", "negative")):
            col.metric(lab.capitalize(), f"{counts.get(lab, 0) / len(df):.0%}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sentiment distribution")
            st.bar_chart(counts)
        if "category" in df:
            with c2:
                st.subheader("Sentiment by category")
                st.bar_chart(pd.crosstab(df["category"], df["sentiment"]))

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
            freq = st.radio("Trend period", ["Month", "Year"], horizontal=True)
            trend = (df.dropna(subset=["date"])
                       .groupby([pd.Grouper(key="date", freq="MS" if freq == "Month" else "YS"), "sentiment"])
                       .size().unstack(fill_value=0))
            if not trend.empty:
                st.subheader(f"Sentiment trend per {freq.lower()}")
                st.line_chart(trend)

        if "stars" in df and df["stars"].notna().any():
            actual = df["stars"].map({1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"})
            st.info(f"Agreement between predicted sentiment and the customer's star rating: "
                    f"{(actual == df['sentiment']).mean():.0%}")

        st.subheader("Most negative feedback")
        show_cols = [c for c in ["text", "stars", "sentiment", "confidence", "category", "date"] if c in df]
        st.dataframe(df[df.sentiment == "negative"].sort_values("confidence", ascending=False)[show_cols].head(10),
                     width="stretch", hide_index=True)

        st.download_button("⬇️ Download results CSV", df[show_cols].to_csv(index=False), "feedback_results.csv", "text/csv")
