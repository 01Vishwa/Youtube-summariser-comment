📊 YouTube Comment Analyzer

A Streamlit app that fetches YouTube video comments and provides summarization, sentiment analysis, keyword extraction, and interactive insights.
Built with the YouTube Data API, Hugging Face transformers, and visualization tools like Matplotlib & WordCloud.

🚀 Features

Fetch comments from any YouTube video using the YouTube Data API.

Sentiment analysis

Hugging Face transformer model (optional).

Rule-based fallback.

Summarization of all comments using BART / T5 (optional).

Keyword analytics (top keywords, bar charts, word cloud).

Interactive exploration:

Filter by sentiment, likes, author.

Search comments by text.

Download filtered results as CSV.

Visual insights:

Sentiment distribution.

Likes histogram.

Comment activity over time.

Video details: title, thumbnail, metadata.

Most engaged authors and highlight of top liked / positive / negative comments.

🛠️ Installation

Clone this repo and install dependencies:

git clone https://github.com/yourusername/youtube-comment-analyzer.git
cd youtube-comment-analyzer
pip install -r requirements.txt

Requirements

Python 3.9+

Streamlit

google-api-python-client

transformers

nltk

wordcloud

matplotlib

pillow

langdetect (optional)

🔑 YouTube API Setup

Go to Google Cloud Console
.

Enable YouTube Data API v3.

Create an API key.

Add it in one of two ways:

Streamlit Secrets:
Create .streamlit/secrets.toml with:

YOUTUBE_API_KEY = "your_api_key_here"


Sidebar input: paste API key directly when running the app.

▶️ Usage

Run the app locally:

streamlit run youtube_comment_analyzer.py


Enter a YouTube video URL or ID.

Adjust options (number of comments, ordering, models).

Click 🚀 Analyze.

Explore summary, charts, keyword analytics, and filters interactively.

📷 Screenshots
Dashboard

(insert screenshot of the app showing summary, charts, and word cloud)

Interactive Explorer

(insert screenshot showing filters and comment search)

📦 Project Structure
youtube_comment_analyzer.py   # Main Streamlit app
requirements.txt              # Dependencies
README.md                     # Documentation

⚡ Notes & Tips

Hugging Face models (sentiment/summarizer) may download on first run — ensure good internet & enough RAM.

For larger comment sets, API quota may be consumed quickly.

langdetect is optional — enable language detection if you want multi-language analysis.

Word cloud and summarizer require enough comment text to generate meaningful results.

🧭 Roadmap / Ideas

🌍 Auto-translate non-English comments before analysis.

🗣 Conversational summarization with LLMs.

📥 Export to Excel / PDF.

📊 More advanced analytics (topic modeling, clustering).

🤝 Contributing

Pull requests and feature requests are welcome!
If you’d like to add improvements (e.g., translations, new visualizations), open an issue or PR.

📜 License

MIT License — feel free to use and modify.
