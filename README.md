# 📊 YouTube Comment Analyzer

A **Streamlit app** that fetches and analyzes YouTube video comments.  
Provides **summarization, sentiment analysis, keyword extraction, word clouds, and interactive insights**.  

Built with:
- [YouTube Data API v3](https://developers.google.com/youtube/v3)
- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [NLTK](https://www.nltk.org/)
- [WordCloud](https://amueller.github.io/word_cloud/)
- [Live Comment Analysis Link](https://youtube-comment-analyser-st.streamlit.app/)

---

## 🚀 Features

- 🔑 **Fetch YouTube comments** using YouTube Data API v3  
- 😀 **Sentiment analysis**  
  - Hugging Face models (DistilBERT, RoBERTa)  
  - Rule-based fallback if models unavailable  
- 📝 **Summarization** of comments (BART / T5)  
- 📊 **Keyword analytics** (top keywords, word cloud)  
- 🔍 **Interactive exploration**  
  - Filter by sentiment, likes, author  
  - Search within comments  
  - Download filtered results as CSV  
- 📈 **Visual insights**  
  - Sentiment distribution  
  - Likes histogram  
  - Comments over time  
- 🏆 **Highlights**  
  - Most liked comment  
  - Top positive / negative comment  
  - Most engaged authors  

---

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/youtube-comment-analyzer.git
cd youtube-comment-analyzer
pip install -r requirements.txt
