# youtube_comment_analyzer.py
# Updated Streamlit app — reorganized UI + extra insights & charts

import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import warnings
from collections import Counter
from io import BytesIO
from datetime import datetime

# YouTube API
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# NLP & visuals
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import requests

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Transformers (optional)
from transformers import pipeline

# optional language detection
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

warnings.filterwarnings("ignore")
st.set_page_config(page_title="YouTube Comment Analyzer", page_icon="📊", layout="wide")

# ---------------------------
# Utilities & caching
# ---------------------------

@st.cache_resource
def download_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

download_nltk()

@st.cache_data(show_spinner=False)
def extract_video_id(url_or_id: str) -> str | None:
    if not url_or_id:
        return None
    url_or_id = url_or_id.strip()
    # If looks like pure id
    if re.fullmatch(r'^[A-Za-z0-9_\-]{11}$', url_or_id):
        return url_or_id
    patterns = [
        r'(?:v=|\/v\/|youtu\.be\/|\/embed\/)([A-Za-z0-9_\-]{11})',
        r'youtube\.com\/shorts\/([A-Za-z0-9_\-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    parts = re.split(r'[\/\?&=]', url_or_id)
    for p in parts[::-1]:
        if re.fullmatch(r'^[A-Za-z0-9_\-]{11}$', p):
            return p
    return None

def build_youtube_client(api_key: str):
    if not api_key:
        raise ValueError("YouTube API key not provided.")
    return build('youtube', 'v3', developerKey=api_key)

def fetch_top_comments(video_id: str, api_key: str, max_comments: int = 10, order: str = "relevance"):
    comments = []
    try:
        youtube = build_youtube_client(api_key)
    except Exception as e:
        st.error(f"Could not build YouTube client: {e}")
        return comments

    next_page_token = None
    fetched = 0
    while fetched < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - fetched),
                pageToken=next_page_token,
                order=order,
                textFormat="plainText"
            )
            response = request.execute()
            items = response.get('items', [])
            for item in items:
                snippet = item['snippet']
                top = snippet.get('topLevelComment', {}).get('snippet', {})
                comments.append({
                    'author': top.get('authorDisplayName', 'Unknown'),
                    'text': top.get('textDisplay', '').strip(),
                    'likes': int(top.get('likeCount', 0)),
                    'published_at': top.get('publishedAt', ''),
                    # total reply count is at top-level snippet
                    'reply_count': int(snippet.get('totalReplyCount', 0))
                })
                fetched += 1
                if fetched >= max_comments:
                    break
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except HttpError as e:
            status = None
            try:
                status = e.resp.status
            except Exception:
                pass
            if status and 500 <= status < 600:
                st.warning(f"Transient API error (status {status}). Retrying...")
                time.sleep(2)
                continue
            st.error(f"YouTube API error: {e}")
            break
        except Exception as e:
            st.error(f"Unexpected error while fetching comments: {e}")
            break
    return comments

@st.cache_data
def get_video_details(video_id: str, api_key: str):
    try:
        youtube = build_youtube_client(api_key)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        items = response.get('items', [])
        if not items:
            return None
        snippet = items[0]['snippet']
        return {
            'title': snippet.get('title', 'Unknown Title'),
            'thumbnail': snippet.get('thumbnails', {}).get('medium', {}).get('url', '')
        }
    except Exception as e:
        st.error(f"Could not fetch video details: {e}")
        return None

# ---------------------------
# Text cleaning & NLP
# ---------------------------

def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text.strip().lower()

@st.cache_resource
def load_sentiment_pipeline(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    try:
        return pipeline("sentiment-analysis", model=model_name)
    except Exception:
        try:
            return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        except Exception:
            return None

def rule_based_sentiment(text: str):
    text = text.lower()
    positive = set(["good","great","awesome","love","excellent","best","amazing","nice","happy","like","fantastic","fun"])
    negative = set(["bad","terrible","hate","awful","worst","dislike","boring","stupid","angry","sad","disappointing","useless"])
    pos_score = sum(1 for w in positive if w in text)
    neg_score = sum(1 for w in negative if w in text)
    if pos_score > neg_score:
        return "Positive"
    elif neg_score > pos_score:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiments(comments: list[dict], sentiment_pipe):
    results = []
    if not comments:
        return results
    if sentiment_pipe is None:
        for c in comments:
            results.append(rule_based_sentiment(preprocess_text(c['text'])[:512]))
        return results
    texts = [c['text'][:512] for c in comments]
    try:
        batch_size = 16
        preds = []
        for i in range(0, len(texts), batch_size):
            slice_texts = texts[i:i+batch_size]
            try:
                out = sentiment_pipe(slice_texts, truncation=True)
            except Exception:
                out = [sentiment_pipe(t[:512])[0] for t in slice_texts]
            preds.extend(out)
        mapping = {
            'POSITIVE': 'Positive',
            'NEGATIVE': 'Negative',
            'NEUTRAL': 'Neutral',
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral',
            'LABEL_2': 'Positive'
        }
        for p in preds:
            label = p.get('label', '')
            results.append(mapping.get(label, label))
    except Exception:
        for c in comments:
            results.append(rule_based_sentiment(preprocess_text(c['text'])[:512]))
    return results

@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception:
        try:
            return pipeline("summarization", model="t5-small")
        except Exception:
            return None

def generate_summary_for_comments(comments: list[dict], summarizer, max_chars=300):
    all_text = " ".join([c.get('text','') for c in comments]).strip()
    if not all_text:
        return "No comment text to summarize."
    if summarizer is None:
        sentences = re.split(r'(?<=[.!?]) +', all_text)
        sentences = sorted(sentences, key=lambda s: -len(s))[:3]
        short = " ".join(sentences)[:max_chars]
        if len(short) >= max_chars:
            short = short[:max_chars].rsplit(' ',1)[0] + "..."
        return short
    # chunk and summarize
    chunk_size = 900
    chunks = []
    if len(all_text) > chunk_size:
        sentences = re.split(r'(?<=[.!?]) +', all_text)
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 <= chunk_size:
                current += s + " "
            else:
                if current:
                    chunks.append(current.strip())
                current = s + " "
        if current:
            chunks.append(current.strip())
    else:
        chunks = [all_text]
    summaries = []
    try:
        for ch in chunks:
            out = summarizer(ch, max_length=150, min_length=30, do_sample=False)
            if out and isinstance(out, list) and out[0].get('summary_text'):
                summaries.append(out[0]['summary_text'])
    except Exception:
        pass
    if summaries:
        combined = " ".join(summaries)
        if len(combined) > max_chars:
            try:
                out = summarizer(combined, max_length=max_chars//2, min_length=30, do_sample=False)
                if out and out[0].get('summary_text'):
                    return out[0]['summary_text']
            except Exception:
                return combined[:max_chars].rsplit(' ',1)[0] + "..."
        return combined
    return all_text[:max_chars].rsplit(' ',1)[0] + "..."

# keyword extraction
def get_top_keywords(comments: list[dict], top_n: int = 15):
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()
    words = []
    for c in comments:
        text = preprocess_text(c.get('text',''))
        toks = [t for t in word_tokenize(text) if t.isalpha() and t not in stop_words and len(t) > 2]
        words.extend(toks)
    counts = Counter(words)
    most = counts.most_common(top_n)
    return pd.DataFrame(most, columns=['word','count'])

def create_wordcloud_image(comments: list[dict]):
    all_text = " ".join(preprocess_text(c.get('text','')) for c in comments)
    if not all_text.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color='white', max_words=150)
    wc.generate(all_text)
    return wc

# ---------------------------
# UI: layout & main
# ---------------------------

# small CSS to improve spacing/card look
st.markdown(
    """
    <style>
    .metric-row { display: flex; gap: 1rem; align-items: center; }
    .card { padding: 1rem; border-radius: 8px; background-color: #fafafa; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
    </style>
    """, unsafe_allow_html=True
)

def main():
    st.title("📊 YouTube Comment Analyzer")
    st.write("Analyze YouTube comments — summarization, sentiment, keyword analytics & interactive exploration.")

    # Sidebar inputs (tidy)
    st.sidebar.header("Inputs & Options")
    api_key_sidebar = st.sidebar.text_input("YouTube API key (optional override)", type="password")
    # default from secrets if present
    api_key = api_key_sidebar.strip() or (st.secrets.get("YOUTUBE_API_KEY") if hasattr(st, "secrets") else None)

    video_url = st.sidebar.text_input("YouTube video URL or ID", placeholder="https://www.youtube.com/watch?v=...")
    max_comments = st.sidebar.number_input("Top comments to fetch", min_value=1, max_value=500, value=50, step=1)
    order = st.sidebar.selectbox("Order", options=["relevance", "time"], index=0)
    # model options
    st.sidebar.markdown("---")
    enable_sentiment_model = st.sidebar.checkbox("Use Hugging Face sentiment model", value=False)
    enable_summarizer = st.sidebar.checkbox("Use Hugging Face summarizer", value=False)
    show_language_detection = st.sidebar.checkbox("Enable language detection", value=True)
    st.sidebar.markdown("---")
    analyze_btn = st.sidebar.button("🚀 Analyze")

    if not analyze_btn:
        st.info("Provide a video URL/ID and press Analyze from the sidebar. You can paste API key in sidebar or use Streamlit secrets.")
        return

    if not video_url:
        st.error("Please enter a YouTube video URL or ID.")
        return
    if not api_key:
        st.error("YouTube API key not found. Add it in the sidebar or in .streamlit/secrets.toml as 'YOUTUBE_API_KEY'.")
        return

    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Could not extract a video ID from the provided URL/ID. Check the input.")
        return

    with st.spinner("Fetching video details..."):
        video_info = get_video_details(video_id, api_key)
        if video_info is None:
            st.error("Couldn't fetch video details. Check API key, video ID or quota.")
            return

    with st.spinner(f"Fetching top {max_comments} comments..."):
        comments = fetch_top_comments(video_id, api_key, max_comments=int(max_comments), order=order)
        if not comments:
            st.warning("No comments found or could not fetch comments.")
            return

    # convert to dataframe
    df = pd.DataFrame(comments)
    if not df.empty:
        # parse published_at as datetime
        try:
            df['published_dt'] = pd.to_datetime(df['published_at'], errors='coerce')
            df['date'] = df['published_dt'].dt.date
        except Exception:
            df['published_dt'] = pd.NaT
            df['date'] = None

    # load models if requested (cached)
    sentiment_pipe = None
    summarizer = None
    if enable_sentiment_model:
        with st.spinner("Loading sentiment model..."):
            sentiment_pipe = load_sentiment_pipeline()
            if sentiment_pipe is None:
                st.warning("Could not load transformer sentiment model — will use rule-based fallback.")
            else:
                st.success("Sentiment model loaded.")
    if enable_summarizer:
        with st.spinner("Loading summarizer..."):
            summarizer = load_summarizer()
            if summarizer is None:
                st.warning("Could not load summarizer pipeline — using extractive fallback.")
            else:
                st.success("Summarizer loaded.")

    # analyze sentiment
    with st.spinner("Analyzing sentiment..."):
        sentiments = analyze_sentiments(comments, sentiment_pipe)
    df['sentiment'] = sentiments + ["Neutral"] * (len(df) - len(sentiments))

    # language detection (optional)
    if show_language_detection and LANGDETECT_AVAILABLE:
        try:
            df['lang'] = df['text'].apply(lambda t: detect(preprocess_text(t)) if t and len(preprocess_text(t)) > 0 else 'unknown')
        except Exception:
            df['lang'] = 'unknown'
    else:
        df['lang'] = 'unknown' if not LANGDETECT_AVAILABLE else df.get('lang', 'unknown')

    # summary
    with st.spinner("Generating summary..."):
        summary_text = generate_summary_for_comments(comments, summarizer)

    # layout: left column for video, right for metrics
    left, right = st.columns([1, 2])
    with left:
        thumb = video_info.get('thumbnail')
        if thumb:
            try:
                r = requests.get(thumb, timeout=5)
                img = Image.open(BytesIO(r.content))
                st.image(img, caption="Video thumbnail", use_container_width=True) # <-- FIX HERE
            except Exception:
                st.info("Thumbnail couldn't be loaded.")
    with right:
        st.subheader(video_info.get('title', 'Video'))
        st.markdown(f"**Video ID:** `{video_id}`")
        st.markdown(f"**Fetched comments:** **{len(df)}** (showing top {min(10, len(df))})")
        # quick metrics in one row
        avg_likes = int(df['likes'].mean()) if not df['likes'].empty else 0
        top_author = df['author'].value_counts().index[0] if not df['author'].empty else 'N/A'
        top_author_count = int(df['author'].value_counts().iloc[0]) if not df['author'].empty else 0
        cols = st.columns(3)
        cols[0].metric("Avg likes", f"{avg_likes}")
        cols[1].metric("Top author", f"{top_author} ({top_author_count})")
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        cols[2].metric("Sentiment sample", ", ".join([f"{k}:{v}" for k, v in sentiment_counts.items()]))

    st.markdown("---")
    # Summary & insights
    st.subheader("📝 Summary")
    st.info(summary_text)

    st.markdown("### Key insights")
    insight_cols = st.columns(3)
    # most liked comment
    most_liked = df.sort_values('likes', ascending=False).head(1)
    if not most_liked.empty:
        ml = most_liked.iloc[0]
        insight_cols[0].write("**Most liked comment**")
        insight_cols[0].write(f"> {ml['text']}")
        insight_cols[0].caption(f"{ml['author']} — {ml['likes']} likes")
    else:
        insight_cols[0].write("No likes info")

    # top positive & negative
    pos_top = df[df['sentiment'] == 'Positive'].sort_values('likes', ascending=False).head(1)
    neg_top = df[df['sentiment'] == 'Negative'].sort_values('likes', ascending=False).head(1)
    insight_cols[1].write("**Top positive**")
    if not pos_top.empty:
        p = pos_top.iloc[0]
        insight_cols[1].write(f"> {p['text']}")
        insight_cols[1].caption(f"{p['author']} — {p['likes']} likes")
    else:
        insight_cols[1].write("No positive comments")

    insight_cols[2].write("**Top negative**")
    if not neg_top.empty:
        n = neg_top.iloc[0]
        insight_cols[2].write(f"> {n['text']}")
        insight_cols[2].caption(f"{n['author']} — {n['likes']} likes")
    else:
        insight_cols[2].write("No negative comments")

    st.markdown("---")

    # Visualizations: sentiment distribution, likes histogram, comments over time
    st.subheader("📈 Visualizations")
    vis1, vis2 = st.columns([1,1])

    # sentiment distribution
    with vis1:
        st.markdown("**Sentiment distribution**")
        sd = df['sentiment'].value_counts().rename_axis('sentiment').reset_index(name='count')
        if not sd.empty:
            st.bar_chart(data=sd.set_index('sentiment'))
        else:
            st.info("No sentiment data")

    # likes histogram
    with vis2:
        st.markdown("**Likes distribution**")
        if not df['likes'].empty:
            fig, ax = plt.subplots(figsize=(4,2.5))
            ax.hist(df['likes'].clip(0, df['likes'].quantile(0.95)), bins=10)
            ax.set_xlabel("Likes (clipped at 95th pct)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("No likes data")

    # comments over time
    st.markdown("**Comments over time**")
    if 'date' in df and df['date'].notna().any():
        times_series = df.groupby('date').size().reset_index(name='count').sort_values('date')
        st.line_chart(data=times_series.set_index('date'))
    else:
        st.info("No timestamped comments available")

    st.markdown("---")
    # Keyword analytics & wordcloud
    st.subheader("📊 Keyword Analytics")
    kw_top_n = st.slider("Top keywords to show", min_value=5, max_value=50, value=15)
    keywords_df = get_top_keywords(comments, top_n=kw_top_n)
    if not keywords_df.empty:
        st.bar_chart(keywords_df.set_index('word')['count'])
        st.dataframe(keywords_df, use_container_width=True, height=200)
    else:
        st.info("No keywords extracted.")

    # Word cloud
    with st.spinner("Generating word cloud..."):
        wc = create_wordcloud_image(comments)
    if wc is not None:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("Word cloud couldn't be generated (insufficient text).")

    st.markdown("---")
    # Top comments table
    st.subheader("Top comments (first 50 shown)")
    display_cols = ['author','text','likes','published_at','reply_count','sentiment','lang']
    display_df = df[display_cols].copy()
    display_df.columns = ['Author','Comment','Likes','Published At','Reply Count','Sentiment','Lang']
    st.dataframe(display_df.head(200), use_container_width=True, height=300)

    # Interactive explorer with filters
    st.markdown("---")
    st.subheader("🔎 Interactive Comment Explorer")
    col_a, col_b, col_c = st.columns([1,1,1])
    sentiment_filter = col_a.multiselect("Filter by sentiment", options=sorted(df['sentiment'].unique().tolist()), default=sorted(df['sentiment'].unique().tolist()))
    min_likes = int(col_b.slider("Minimum likes", min_value=0, max_value=int(df['likes'].max()) if not df['likes'].empty else 0, value=0))
    author_search = col_c.text_input("Author contains", "")

    filtered = df[
        (df['sentiment'].isin(sentiment_filter)) &
        (df['likes'] >= min_likes) &
        (df['author'].str.lower().str.contains(author_search.lower()))
    ]
    st.write(f"Showing {len(filtered)} / {len(df)} comments")
    if not filtered.empty:
        st.dataframe(filtered[['author','text','likes','published_at','reply_count','sentiment','lang']], use_container_width=True, height=300)
        csv_all = filtered.to_csv(index=False)
        st.download_button("Download filtered comments (CSV)", csv_all, file_name="filtered_comments.csv", mime="text/csv")
    else:
        st.info("No comments match the filters.")

    # Search box for comments
    st.markdown("---")
    st.subheader("🔍 Search comments")
    search_term = st.text_input("Search for comments containing:")
    if search_term:
        matches = df[df['text'].str.lower().str.contains(search_term.lower(), na=False)]
        st.write(f"Found {len(matches)} comment(s) containing '{search_term}'")
        if not matches.empty:
            st.dataframe(matches[['author','text','likes','published_at','sentiment','lang']], use_container_width=True)
            st.download_button("Download matches CSV", matches.to_csv(index=False), file_name="matching_comments.csv", mime="text/csv")
        else:
            st.info("No matches")

    # Insights: most engaged authors
    st.markdown("---")
    st.subheader("🏆 Most engaged authors")
    if not df.empty:
        authors = df.groupby('author').agg(count=('text','size'), total_likes=('likes','sum')).sort_values(['count','total_likes'], ascending=False).reset_index()
        st.dataframe(authors.head(20), use_container_width=True, height=200)
    else:
        st.info("No author data")

    # Footer tips
    st.markdown("---")
    st.caption("Tip: For heavy usage enable model caching and consider running on a machine with enough RAM — HF models may download when enabled. If you want auto-translation of non-English comments or advanced conversational summarization, I can add that as an optional feature.")

if __name__ == "__main__":
    main()