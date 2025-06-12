import streamlit as st
import nltk
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import googleapiclient.discovery
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np # For potential use in calculations like std dev or handling NaN
# import plotly.graph_objects as go # Removed as plot_waveform and plot_spider_graph are removed

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI # Changed from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate # Not using custom prompt for basic version
from langchain.memory import ConversationBufferMemory # May not be used if switching to RetrievalQA without memory
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter # Not using for now
from transformers import pipeline # For toxicity detection

# LangGraph and typing imports
from typing import TypedDict, List, Annotated, Any # Added Any
from langgraph.graph import StateGraph, END

# scikit-learn imports for clustering and dimensionality reduction
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# --- Graph State Definition ---
class GraphState(TypedDict):
    youtube_url: str
    comments_df: pd.DataFrame = None # Allow None initially
    # FAISS retriever can be complex, using Any for now
    vector_store_retriever: Any = None
    analysis_summary: str = None # Allow None initially
    error_message: str = None # Allow None initially
    # Configuration for API keys can be implicitly handled by cached functions like get_llm()
    cluster_summaries: dict = None # For LLM generated cluster themes

# --- Clustering and Dimensionality Reduction Functions ---

def perform_kmeans_clustering(embeddings_array: np.ndarray, num_clusters: int) -> np.ndarray:
    """Performs K-Means clustering on embeddings."""
    if embeddings_array is None or embeddings_array.shape[0] < num_clusters:
        st.warning(f"Not enough data points ({embeddings_array.shape[0] if embeddings_array is not None else 0}) for {num_clusters} clusters. Skipping K-Means.")
        return np.array([-1] * (embeddings_array.shape[0] if embeddings_array is not None else 0)) # Return -1 for all labels

    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embeddings_array)
        return cluster_labels
    except Exception as e:
        st.error(f"Error during K-Means clustering: {e}")
        return np.array([-1] * embeddings_array.shape[0]) # Return -1 for all labels on error

def perform_tsne(embeddings_array: np.ndarray, n_components=2) -> np.ndarray:
    """Performs t-SNE dimensionality reduction."""
    if embeddings_array is None or embeddings_array.shape[0] < 2 : # t-SNE needs at least 2 samples
        st.warning("Not enough data points for t-SNE. Skipping.")
        # Return an array of NaNs with the correct shape if it's 2D, otherwise handle appropriately
        return np.full((embeddings_array.shape[0] if embeddings_array is not None else 0, n_components), np.nan)

    try:
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, embeddings_array.shape[0] - 1)) # Perplexity must be less than n_samples
        tsne_results = tsne.fit_transform(embeddings_array)
        return tsne_results
    except Exception as e:
        st.error(f"Error during t-SNE: {e}")
        return np.full((embeddings_array.shape[0], n_components), np.nan) # Return NaNs on error

def summarize_cluster_themes(df: pd.DataFrame, llm, num_clusters: int, comments_per_cluster: int = 3) -> dict:
    """Generates thematic summaries for each comment cluster using an LLM."""
    if llm is None:
        st.warning("LLM not available, skipping cluster summarization.")
        return {}
    if 'cluster_label' not in df.columns or 'cleaned_text' not in df.columns:
        st.warning("Required columns for cluster summarization are missing.")
        return {}

    cluster_themes = {}
    st.write("✍️ Generating cluster theme summaries with LLM...")
    for i in range(num_clusters):
        cluster_df = df[df['cluster_label'] == i]
        if cluster_df.empty:
            cluster_themes[i] = "No comments in this cluster."
            continue

        # Sample comments, ensuring not to sample more than available
        sample_size = min(comments_per_cluster, len(cluster_df))
        sampled_comments = cluster_df['cleaned_text'].sample(n=sample_size, random_state=42).tolist()

        if not sampled_comments:
            cluster_themes[i] = "Not enough comments to sample for summarization."
            continue

        prompt = f"""The following are {len(sampled_comments)} example comments from a discussion cluster related to a YouTube video.
        What is the main theme or topic of this cluster? Please provide a concise theme name (3-5 words).
        Comments:
        {'- ' + '\n- '.join(sampled_comments)}

        Main Theme:"""

        try:
            with st.spinner(f"Summarizing theme for cluster {i}..."):
                response = llm.invoke(prompt)
                theme = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            cluster_themes[i] = theme
        except Exception as e:
            st.error(f"Error summarizing theme for cluster {i}: {e}")
            cluster_themes[i] = "Error generating theme."

    st.write("✅ Cluster themes summarized.")
    return cluster_themes

# --- LangGraph Node Functions ---

def cluster_comments_node(state: GraphState) -> GraphState:
    """Performs comment clustering, t-SNE, and generates LLM summaries for clusters."""
    st.write("📊 Performing comment clustering and t-SNE analysis...")
    comments_df = state.get("comments_df")

    if comments_df is None or comments_df.empty or 'embedding' not in comments_df.columns or comments_df['embedding'].isnull().all():
        st.warning("No embeddings available for clustering. Skipping.")
        # Ensure necessary columns exist even if empty/default
        if comments_df is not None:
            comments_df['cluster_label'] = -1
            comments_df['tsne_x'] = np.nan
            comments_df['tsne_y'] = np.nan
        return {**state, "comments_df": comments_df, "cluster_summaries": {}}

    # Convert list of embeddings to NumPy array
    # Filter out rows where 'embedding' might be None if some failed
    valid_embeddings_df = comments_df.dropna(subset=['embedding'])
    if valid_embeddings_df.empty:
        st.warning("No valid embeddings found for clustering after filtering. Skipping.")
        comments_df['cluster_label'] = -1
        comments_df['tsne_x'] = np.nan
        comments_df['tsne_y'] = np.nan
        return {**state, "comments_df": comments_df, "cluster_summaries": {}}

    embeddings_array = np.array(valid_embeddings_df['embedding'].tolist())

    # Determine number of clusters (simple heuristic)
    num_clusters = max(2, min(5, len(valid_embeddings_df) // 10)) # Aim for 2-5 clusters
    if len(valid_embeddings_df) < num_clusters: # Not enough samples for defined clusters
        num_clusters = max(1, len(valid_embeddings_df)) # Adjust if fewer samples than min clusters
        if num_clusters == 1 and len(valid_embeddings_df) < 2: # KMeans needs at least 1 sample, but tSNE needs more
            st.warning("Very few comments for meaningful clustering. Assigning to single cluster.")
            # Handle single comment case: tSNE will fail.
            comments_df.loc[valid_embeddings_df.index, 'cluster_label'] = 0
            comments_df['tsne_x'] = 0.0 # Placeholder for t-SNE
            comments_df['tsne_y'] = 0.0 # Placeholder for t-SNE
            llm = get_llm()
            cluster_summaries = summarize_cluster_themes(comments_df[comments_df['cluster_label'] == 0], llm, 1) if llm else {"0": "Single comment cluster"}
            return {**state, "comments_df": comments_df, "cluster_summaries": cluster_summaries}


    cluster_labels = perform_kmeans_clustering(embeddings_array, num_clusters)
    # Assign labels back to the original DataFrame at corresponding valid indices
    comments_df.loc[valid_embeddings_df.index, 'cluster_label'] = cluster_labels
    # Fill non-clustered rows (if any had missing embeddings) with -1
    if 'cluster_label' not in comments_df.columns: # If column wasn't created due to all embeddings missing
        comments_df['cluster_label'] = -1
    else:
        comments_df['cluster_label'] = comments_df['cluster_label'].fillna(-1).astype(int)


    tsne_results = perform_tsne(embeddings_array)
    comments_df.loc[valid_embeddings_df.index, 'tsne_x'] = tsne_results[:, 0]
    comments_df.loc[valid_embeddings_df.index, 'tsne_y'] = tsne_results[:, 1]
    # Fill non-tSNE'd rows
    if 'tsne_x' not in comments_df.columns: comments_df['tsne_x'] = np.nan
    else: comments_df['tsne_x'] = comments_df['tsne_x'].fillna(np.nan)
    if 'tsne_y' not in comments_df.columns: comments_df['tsne_y'] = np.nan
    else: comments_df['tsne_y'] = comments_df['tsne_y'].fillna(np.nan)

    llm = get_llm() # Cached
    cluster_summaries = {}
    if llm and num_clusters > 0 and -1 not in cluster_labels : # Only summarize if clustering happened and LLM is available
         # Ensure that num_clusters passed to summarize_cluster_themes is correct if some clusters are empty
        actual_num_clusters_with_assignments = len(np.unique(cluster_labels[cluster_labels != -1]))
        if actual_num_clusters_with_assignments > 0:
             cluster_summaries = summarize_cluster_themes(comments_df[comments_df['cluster_label'] != -1], llm, actual_num_clusters_with_assignments)
        else:
            st.info("No valid clusters formed for summarization.")
    elif not llm:
        st.info("LLM not available, skipping cluster theme summarization.")

    st.write("✅ Clustering and t-SNE complete.")
    return {**state, "comments_df": comments_df, "cluster_summaries": cluster_summaries}

def fetch_youtube_comments_node(state: GraphState) -> GraphState:
    """Fetches YouTube comments and updates the state."""
    st.write("▶️ Fetching YouTube comments...")
    youtube_url = state.get("youtube_url")
    if not youtube_url:
        return {**state, "error_message": "YouTube URL is missing."}

    try:
        comments_df = fetch_comments(youtube_url, max_comments=1500) # Using existing function
        if comments_df is not None and not comments_df.empty:
            st.write(f"📊 Fetched {len(comments_df)} comments.")
            return {**state, "comments_df": comments_df, "error_message": None}
        else:
            st.write("⚠️ No comments fetched or an error occurred during fetching.")
            return {**state, "comments_df": None, "error_message": "Failed to fetch comments or no comments found."}
    except Exception as e:
        st.error(f"Exception in fetch_youtube_comments_node: {e}")
        return {**state, "comments_df": None, "error_message": str(e)}

def preprocess_and_analyze_node(state: GraphState) -> GraphState:
    """Preprocesses comments, performs sentiment & toxicity analysis, and creates vector store."""
    st.write("🔄 Preprocessing and analyzing comments...")
    comments_df = state.get("comments_df")
    if comments_df is None or comments_df.empty:
        st.write("⚠️ No comments DataFrame to process.")
        return {**state, "error_message": "No comments to process."}

    try:
        # 1. Text cleaning & initial sentiment (VADER) - already part of main logic that will be wrapped
        # Ensure 'text' column is string type
        comments_df['text'] = comments_df['text'].astype(str)
        sentiment_results = comments_df['text'].apply(clean_and_analyze_text)
        comments_df['cleaned_text'] = sentiment_results.apply(lambda x: x[0])
        comments_df['sentiment_scores'] = sentiment_results.apply(lambda x: x[1])
        comments_df['sentiment_compound'] = comments_df['sentiment_scores'].apply(lambda x: x['compound'])

        # 2. Timestamp Parsing and Sorting
        try:
            comments_df['published_at'] = pd.to_datetime(comments_df['published_at'])
            comments_df = comments_df.sort_values(by='published_at', ascending=True)
        except Exception as e:
            st.error(f"Error parsing timestamps in graph node: {e}")
            # Continue without timestamp features if parsing fails for some comments
            # Or return {**state, "error_message": f"Timestamp parsing error: {e}"}

        # 3. Feature 1: Sentiment Timeline calculations
        if 'sentiment_compound' in comments_df.columns and 'published_at' in comments_df.columns:
            comments_df = calculate_sliding_window_sentiment(comments_df)
            if 'sentiment_sliding_avg' in comments_df.columns:
                comments_df = detect_emotion_drift(comments_df)
            else: # ensure column exists
                comments_df['is_drift_point'] = False
        else: # ensure columns exist
            comments_df['sentiment_sliding_avg'] = np.nan
            comments_df['is_drift_point'] = False


        # 4. Feature 3: Toxicity Analysis
        toxicity_classifier = get_toxicity_pipeline() # Cached
        if toxicity_classifier:
            if 'cleaned_text' in comments_df.columns:
                comments_df = analyze_comment_toxicity(comments_df, toxicity_classifier)
            else:
                comments_df['is_toxic'] = False
                comments_df['toxicity_score'] = 0.0
        else:
            comments_df['is_toxic'] = False
            comments_df['toxicity_score'] = 0.0
            st.info("Toxicity analysis model not loaded, skipping toxicity in graph.")

        # 5. Feature 2: Vector Store Creation & Storing Raw Embeddings
        hf_embeddings_model = get_hf_embeddings() # Cached
        retriever = None
        if hf_embeddings_model and 'cleaned_text' in comments_df.columns and not comments_df['cleaned_text'].empty:
            comment_texts = comments_df['cleaned_text'].tolist()
            try:
                with st.spinner("Generating comment embeddings for clustering/vector store..."):
                    raw_embeddings = hf_embeddings_model.embed_documents(comment_texts)
                comments_df['embedding'] = raw_embeddings # Store raw embeddings
                # Create vector store for RAG
                # Note: create_comment_vector_store uses FAISS.from_texts which re-embeds.
                # This is slightly inefficient as we already have raw_embeddings.
                # For now, keeping it simple. A future optimization could be FAISS.from_embeddings.
                retriever = create_comment_vector_store(comments_df, _embeddings_model=hf_embeddings_model)
            except Exception as e:
                st.error(f"Error generating embeddings or vector store: {e}")
                comments_df['embedding'] = None # Ensure column exists but is None
        else:
            comments_df['embedding'] = None # Ensure column exists but is None

        st.write("✅ Analysis complete.")
        return {**state, "comments_df": comments_df, "vector_store_retriever": retriever, "error_message": None}
    except Exception as e:
        st.error(f"Exception in preprocess_and_analyze_node: {e}")
        return {**state, "error_message": str(e)}

def generate_insights_node(state: GraphState) -> GraphState:
    """Generates insights from the analyzed comments using an LLM."""
    st.write("💡 Generating insights...")
    comments_df = state.get("comments_df")
    # vector_store_retriever = state.get("vector_store_retriever") # Not directly used for summary here

    if comments_df is None or comments_df.empty:
        return {**state, "analysis_summary": "No comments data to generate insights from."}

    llm = get_llm() # Cached
    if not llm:
        return {**state, "analysis_summary": "LLM not available to generate insights."}

    try:
        # Prepare a summary of data for the prompt
        num_comments = len(comments_df)
        avg_sentiment = comments_df['sentiment_compound'].mean() if 'sentiment_compound' in comments_df else 'N/A'
        num_toxic = comments_df['is_toxic'].sum() if 'is_toxic' in comments_df else 'N/A'

        # Sample some comments (e.g., a few positive, a few negative, a few toxic if available)
        # For brevity, we'll just use aggregated stats in the prompt.
        # A more advanced version could sample comments or use the RAG retriever for themes.

        prompt_text = f"""
        Analyze the following YouTube comments data and provide a concise summary of viewer reactions.
        Total Comments: {num_comments}
        Average Sentiment (Compound Score): {avg_sentiment:.2f} (where >0 is positive, <0 is negative)
        Number of Comments Flagged as Toxic: {num_toxic}

        Key themes or common topics mentioned in the comments might be (if you can infer any from general knowledge of YouTube comments):
        [Content of the video itself, opinions, discussions between users, spam/irrelevant content]

        Based on this data, what is the overall sentiment? Are there significant toxicity concerns?
        What might be the general public opinion or reaction reflected in these comments?
        Provide a brief, bullet-point summary.
        """

        response = llm.invoke(prompt_text)
        summary = response.content # Assuming response object has a 'content' attribute for ChatOpenAI

        st.write("✅ Insights generated.")
        return {**state, "analysis_summary": summary}
    except Exception as e:
        st.error(f"Exception in generate_insights_node: {e}")
        return {**state, "analysis_summary": f"Error generating insights: {e}"}

# --- LangGraph Construction ---

@st.cache_resource
def get_compiled_graph():
    """Constructs and compiles the LangGraph StateGraph."""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("fetch_comments", fetch_youtube_comments_node)
    graph.add_node("preprocess_analyze", preprocess_and_analyze_node)
    graph.add_node("cluster_comments", cluster_comments_node) # New node
    graph.add_node("generate_insights", generate_insights_node)

    # Define edges
    graph.set_entry_point("fetch_comments")

    # Conditional edges
    def should_continue(state: GraphState, from_node: str):
        if state.get("error_message"):
            st.error(f"Stopping graph due to error after {from_node}: {state.get('error_message')}")
            return END
        if from_node == "fetch_comments" and (state.get("comments_df") is None or state.get("comments_df").empty):
            st.warning("No comments fetched, stopping workflow.")
            return END
        return True # Continue to next node if no error

    graph.add_conditional_edges(
        "fetch_comments",
        lambda state: "preprocess_analyze" if should_continue(state, "fetch_comments") != END else END,
        {"preprocess_analyze": "preprocess_analyze", END: END}
    )
    graph.add_conditional_edges(
        "preprocess_analyze",
        lambda state: "cluster_comments" if should_continue(state, "preprocess_analyze") != END else END,
        {"cluster_comments": "cluster_comments", END: END}
    )
    graph.add_conditional_edges(
        "cluster_comments",
        lambda state: "generate_insights" if should_continue(state, "cluster_comments") != END else END,
        {"generate_insights": "generate_insights", END: END}
    )
    graph.add_edge("generate_insights", END)

    # Compile the graph
    try:
        app = graph.compile()
        st.success("Workflow graph compiled successfully.")
        return app
    except Exception as e:
        st.error(f"Error compiling workflow graph: {e}")
        return None

# --- LangChain Setup Functions ---

@st.cache_resource # Cache the embeddings model
def get_hf_embeddings():
    """Initializes HuggingFace embeddings model."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        return embeddings
    except Exception as e:
        st.error(f"Error initializing HuggingFace embeddings: {e}")
        return None

@st.cache_resource # Cache the LLM client
def get_llm():
    """Initializes ChatOpenAI LLM."""
    openai_api_key = None
    if 'OPENAI_API_KEY' in st.secrets:
        openai_api_key = st.secrets['OPENAI_API_KEY']

    if not openai_api_key:
        st.warning("OpenAI API Key not found in st.secrets. AI query functionality will be disabled.")
        return None
    try:
        llm = ChatOpenAI(api_key=openai_api_key, temperature=0.7) # Initialize with key
        return llm
    except Exception as e:
        st.error(f"Error initializing OpenAI LLM: {e}")
        return None

@st.cache_resource(show_spinner="Building comment knowledge base...") # Cache the vector store
def create_comment_vector_store(comments_df, _embeddings_model): # _embeddings_model to satisfy cache
    """Creates a FAISS vector store from comment texts."""
    if comments_df.empty or 'cleaned_text' not in comments_df:
        st.warning("No comments available or 'cleaned_text' column missing for AI query setup.")
        return None

    texts = comments_df['cleaned_text'].tolist()

    # FAISS typically requires at least 1 vector.
    if not texts:
        st.warning("No valid text found in comments to build knowledge base.")
        return None

    try:
        # Get the actual embeddings model from the cached function
        embeddings_model = get_hf_embeddings()
        if not embeddings_model:
            st.error("Embeddings model not available for vector store creation.")
            return None

        vector_store = FAISS.from_texts(texts, embeddings_model)
        return vector_store.as_retriever() # Return as retriever
    except Exception as e:
        st.error(f"Error creating FAISS vector store: {e}")
        return None

@st.cache_resource(show_spinner="Initializing AI Query Engine...")
def setup_rag_chain(_retriever, _llm): # Use _ to indicate params are for cache dependency
    """Sets up the RetrievalQA chain."""
    retriever = _retriever # get actual retriever from cached resource if passed directly
    llm = get_llm() # get actual llm

    if not retriever or not llm:
        st.warning("Retriever or LLM not available. Cannot setup RAG chain.")
        return None

    try:
        # Using a simple chain type for now.
        # "stuff": Uses all of the text from the documents in the prompt. Good for small number of docs.
        # "map_reduce": Summarizes/maps each document, then reduces the summaries.
        # "refine": Iteratively refines an answer by looking at more documents.
        # "map_rerank": Ranks documents and then uses the best one for a direct answer.
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False # Can be True if we want to show sources
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error setting up RAG chain: {e}")
        return None

# --- Toxicity Analysis Functions ---

@st.cache_resource(show_spinner="Loading toxicity detection model...")
def get_toxicity_pipeline():
    """Initializes the toxicity detection pipeline from transformers."""
    try:
        # Using a smaller, distilled model for potentially faster performance if available
        # Alternatives: "unitary/toxic-bert", "martin-ha/toxic-comment-model"
        # For this example, let's stick to a known general one.
        toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", tokenizer="unitary/toxic-bert")
        return toxicity_classifier
    except Exception as e:
        st.error(f"Error initializing toxicity detection model: {e}")
        return None

def analyze_comment_toxicity(df, classifier, text_column='cleaned_text', toxicity_threshold=0.7):
    """
    Analyzes comments for toxicity using the provided Hugging Face pipeline.
    Adds 'is_toxic' and 'toxicity_score' columns to the DataFrame.
    """
    if df.empty or text_column not in df:
        st.warning("DataFrame is empty or missing text column for toxicity analysis.")
        df['is_toxic'] = False
        df['toxicity_score'] = 0.0
        return df

    if classifier is None:
        st.error("Toxicity classifier not available for analysis.")
        df['is_toxic'] = False
        df['toxicity_score'] = 0.0
        return df

    # Ensure text column is list of strings
    texts_to_analyze = df[text_column].astype(str).tolist()

    toxicity_results = []
    toxicity_scores = []

    # Batch processing could be faster if the pipeline supports it well for many comments.
    # For now, iterating for clarity and individual error handling (though pipeline might handle batch errors too).
    with st.spinner(f"Analyzing {len(texts_to_analyze)} comments for toxicity..."):
        try:
            # The pipeline returns a list of dictionaries, e.g., [{'label': 'toxic', 'score': 0.98}] or [{'label': 'LABEL_1', 'score': 0.98}]
            # The specific label for toxicity depends on the model. 'unitary/toxic-bert' outputs labels like 'toxic', 'severe_toxic', etc.
            # We'll consider any "toxic" related label with high confidence as toxic.
            # For simplicity, let's assume the primary positive class is 'toxic' or if not, the one with highest score if it's a multi-label output.
            # unitary/toxic-bert returns a list of dicts for each input text, each dict for a label.
            # e.g. for a single text: [[{'label': 'toxic', 'score': 0.005779 toxic}, {'label': 'severe_toxic', 'score': 0.0003 severe_toxic}, ...]]
            # This model might be multi-label. Let's simplify for the first pass.
            # A simpler approach is to use a model that directly outputs a "toxic" vs "non-toxic" label.
            # Let's use a model that provides a 'toxic' label directly or check its output structure.
            # The default for pipeline("text-classification", model="unitary/toxic-bert") should be fine.
            # It returns a list of lists of dicts if texts_to_analyze is a list.

            # Let's process one by one to manage complex output structure initially
            for text in texts_to_analyze:
                if not text or text.isspace(): # Skip empty or whitespace-only strings
                    toxicity_results.append(False)
                    toxicity_scores.append(0.0)
                    continue

                result = classifier(text)
                # Expected output for unitary/toxic-bert for a single string:
                # [{'label': 'toxic', 'score': 0.9...}, {'label': 'severe_toxic', ...}]
                # We are interested in the 'toxic' label primarily.
                is_comment_toxic = False
                comment_toxicity_score = 0.0
                if isinstance(result, list) and result:
                    for label_score_pair in result:
                        if label_score_pair['label'].lower() == 'toxic' and label_score_pair['score'] > toxicity_threshold:
                            is_comment_toxic = True
                            comment_toxicity_score = label_score_pair['score']
                            break # Found primary toxic label
                        # Fallback if 'toxic' label isn't the main one, or to capture highest toxic score.
                        if 'toxic' in label_score_pair['label'].lower() and label_score_pair['score'] > comment_toxicity_score :
                             comment_toxicity_score = label_score_pair['score'] # get highest score among toxic related labels
                             if comment_toxicity_score > toxicity_threshold:
                                 is_comment_toxic = True


                toxicity_results.append(is_comment_toxic)
                toxicity_scores.append(comment_toxicity_score)

        except Exception as e:
            st.error(f"Error during batch toxicity analysis: {e}")
            # Fill remaining with non-toxic if batch fails midway
            remaining_count = len(texts_to_analyze) - len(toxicity_results)
            toxicity_results.extend([False] * remaining_count)
            toxicity_scores.extend([0.0] * remaining_count)

    df['is_toxic'] = toxicity_results
    df['toxicity_score'] = toxicity_scores
    return df

# Initialize NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')

#Scrap video id
def get_video_id(song_input):
    if "youtube.com" in song_input or "youtu.be" in song_input:
        video_id_match = re.search(r"(?<=v=)[a-zA-Z0-9_-]+(?=&|\s|$)", song_input)
        if video_id_match:
            return video_id_match.group(0)
        else:
            st.warning("Invalid YouTube link.")
            return None
    else:
        try:
            search_query = f"https://www.youtube.com/results?search_query={song_input.replace(' ', '+')}+audio"
            response = requests.get(search_query)
            soup = BeautifulSoup(response.text, 'html.parser')
            search_results = soup.find_all('a', class_='yt-uix-tile-link')
            if search_results:
                video_id = search_results[0]['href'][9:]
                return video_id
            else:
                st.warning("Could not find video ID from search.")
                return None
        except Exception as e:
            st.error(f"Error fetching video ID: {e}")
            return None

#Scrap the comments for video
def fetch_comments_with_api(video_id, max_comments=1500):
    try:
        api_service_name = "youtube"
        api_version = "v3"

        api_key = None
        if 'DEVELOPER_KEY' in st.secrets:
            api_key = st.secrets['DEVELOPER_KEY']

        if not api_key:
            st.error("YouTube API Key (DEVELOPER_KEY) not configured. Please set it in Streamlit secrets (e.g., secrets.toml).")
            return None

        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey=api_key)

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100)
        )
        response = request.execute()

        comments = []

        while len(comments) < max_comments and 'items' in response:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': comment['authorDisplayName'],
                    'published_at': comment['publishedAt'],
                    'like_count': comment['likeCount'],
                    'text': comment['textDisplay']
                })

            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(max_comments - len(comments), 100),
                    pageToken=response['nextPageToken']
                )
                response = request.execute()
            else:
                break

        return pd.DataFrame(comments)
    except Exception as e:
        st.error("Error fetching comments with API:", e)
        return None

#Fetch the comments
def fetch_comments(song_input, max_comments=500):
    video_id = get_video_id(song_input)
    if video_id:
        df_api = fetch_comments_with_api(video_id, max_comments)
        if df_api is not None:
            return df_api
        else:
            st.error("Failed to fetch comments with API.")
            return None
    else:
        st.warning("Song not found or unable to fetch video ID.")
        return None

#Text cleaning and preprocessing
def clean_and_analyze_text(text):
    cleaned_text = re.sub(r'<[^>]*>', '', text)
    cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_text)
    cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')
    cleaned_text = cleaned_text.lower()
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    cleaned_text = re.sub(r'[^a-zA-Z]', ' ', cleaned_text)
    cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stopwords.words('english')])
    ps = PorterStemmer()
    cleaned_text = ' '.join([ps.stem(word) for word in cleaned_text.split()])
    lemmatizer = nltk.stem.WordNetLemmatizer()
    cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in cleaned_text.split()])

    # Sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(cleaned_text)
    return cleaned_text, sentiment_scores

def calculate_sliding_window_sentiment(df, window_size_minutes=60, sentiment_col='sentiment_compound', time_col='published_at'):
    """
    Calculates the sliding window average of sentiment scores.
    Uses a time-based window.
    """
    if df.empty or time_col not in df or sentiment_col not in df:
        st.warning("DataFrame is empty or missing required columns for sliding window calculation.")
        return df

    # Ensure the time column is datetime and set as index
    temp_df = df.set_index(pd.to_datetime(df[time_col]))

    # Calculate rolling mean. min_periods=1 means it will output a value even if only one data point is in the window.
    window = f'{window_size_minutes}T'
    temp_df[f'{sentiment_col}_sliding_avg'] = temp_df[sentiment_col].rolling(window=window, min_periods=1).mean()

    # Merge the result back to the original DataFrame
    # Preserve the original index of df
    df = df.merge(temp_df[[f'{sentiment_col}_sliding_avg']], left_index=True, right_index=True, how='left')
    df.rename(columns={f'{sentiment_col}_sliding_avg': 'sentiment_sliding_avg'}, inplace=True) # Ensure consistent column name
    return df

def detect_emotion_drift(df, sentiment_col='sentiment_sliding_avg', threshold_factor=1.5):
    """
    Detects emotion drift points based on changes in sliding window average sentiment.
    """
    if df.empty or sentiment_col not in df:
        st.warning("DataFrame is empty or missing sentiment column for drift detection.")
        df['is_drift_point'] = False # Add column even if empty
        return df

    df['sentiment_diff'] = df[sentiment_col].diff()

    # Drop NA for diff calculation (first row will be NaN)
    valid_diffs = df['sentiment_diff'].dropna()
    if valid_diffs.empty:
        df['is_drift_point'] = False
        return df

    drift_threshold = valid_diffs.std() * threshold_factor

    # A drift point is where the absolute difference exceeds the threshold
    df['is_drift_point'] = df['sentiment_diff'].abs() > drift_threshold

    # The first row will have NaN for diff, so it's not a drift point by this definition
    df.loc[df.index[0], 'is_drift_point'] = False

    return df

def plot_sentiment_timeline(df, time_col='published_at', sentiment_col='sentiment_compound', avg_sentiment_col='sentiment_sliding_avg', drift_col='is_drift_point'):
    """
    Plots the sentiment timeline with individual comment sentiments, sliding window average, and drift points.
    """
    if df.empty or time_col not in df or sentiment_col not in df or avg_sentiment_col not in df or drift_col not in df:
        st.warning("DataFrame is empty or missing required columns for plotting sentiment timeline.")
        return None # Return None if plot cannot be generated

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot individual comment sentiments (as scatter)
    ax.scatter(df[time_col], df[sentiment_col], label='Individual Comment Sentiment', alpha=0.5, s=10, color='gray')

    # Plot sliding window average sentiment
    ax.plot(df[time_col], df[avg_sentiment_col], label='Sliding Window Average Sentiment', color='blue', linewidth=2)

    # Highlight drift points
    drift_points = df[df[drift_col]]
    if not drift_points.empty:
        ax.scatter(drift_points[time_col], drift_points[avg_sentiment_col], color='red', marker='o', s=100, label='Emotion Drift Point')

    ax.set_title('Sentiment Timeline and Emotion Drift')
    ax.set_xlabel('Time (Published At)')
    ax.set_ylabel('Sentiment Score (Compound)')
    ax.legend()
    ax.grid(True)

    # Improve date formatting on x-axis
    plt.xticks(rotation=45)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    return fig

# Streamlit App
def main():
    st.image("image/top_image.jpg")
    st.title("YT Sentilyzer - YouTube Comment Intelligence Hub")
    name = st.text_input("Enter your name:")

    if name:
        st.write(f"Hello, {name}!")

        st.write("Welcome! Please enter the YouTube video link you'd like to analyze.")
        youtube_link = st.text_input("Enter YouTube link:")

        if youtube_link:
            st.write(f"Analyzing YouTube link: {youtube_link}")

            # --- LangGraph Workflow Execution ---
            compiled_graph = get_compiled_graph()
            if compiled_graph:
                initial_state = GraphState(
                    youtube_url=youtube_link,
                    comments_df=None,
                    vector_store_retriever=None,
                    analysis_summary=None,
                    error_message=None
                )

                # Use a spinner for the entire graph execution
                with st.spinner("Running automated analysis workflow... Please wait."):
                    final_state = compiled_graph.invoke(initial_state)

                st.markdown("---") # Separator
                st.markdown("## Automated Analysis Insights")
                if final_state.get("error_message"):
                    st.error(f"Workflow error: {final_state['error_message']}")
                if final_state.get("analysis_summary"):
                    st.markdown(final_state["analysis_summary"])
                else:
                    st.info("No automated analysis summary was generated.")
                st.markdown("---") # Separator

                # Retrieve the processed data from the final state for other features
                data = final_state.get("comments_df") # 'data' is the variable name used by subsequent sections

            else: # compiled_graph is None
                st.error("Automated analysis workflow could not be initialized.")
                data = None # Ensure data is None so other sections don't run or show errors

            if data is not None and not data.empty:
                # The following sections (Top comments, keyword search, plots, RAG, Toxicity summary)
                # will now use the 'data' DataFrame that has been processed by the graph.
                # Most of the direct data processing calls in main() have been moved into graph nodes.

                # Ensure 'is_toxic' and 'toxicity_score' columns exist if toxicity analysis was supposed to run
                # (It should be added by preprocess_and_analyze_node)
                if 'is_toxic' not in data.columns: data['is_toxic'] = False
                if 'toxicity_score' not in data.columns: data['toxicity_score'] = 0.0
                if 'is_drift_point' not in data.columns: data['is_drift_point'] = False
                if 'sentiment_sliding_avg' not in data.columns: data['sentiment_sliding_avg'] = np.nan


                # Displaying top comments (modified to show toxicity) - This section uses 'data'
                st.write("### Top/Recent Comments (withToxicity Info):")

                # The data processing steps previously here are now inside the graph nodes.
                # The 'data' DataFrame from final_state should have:
                # - cleaned_text, sentiment_compound, published_at (datetime), is_toxic, toxicity_score,
                # - sentiment_sliding_avg, is_drift_point
                # - vector_store_retriever is also in final_state but not directly used by sections below, RAG uses it.

                comments_to_display_df = pd.DataFrame() # Create an empty df to hold comments for display
                option = st.radio("Select comments to display:", ["All (Recent 5)", "Positive (Top 5)", "Negative (Top 5)", "Neutral (Top 5)"], index=0)

                temp_display_df = data.copy() # Work with a copy for filtering for display

                if option == "Positive (Top 5)":
                    comments_to_display_df = temp_display_df[temp_display_df['sentiment_compound'] > 0].nlargest(5, 'sentiment_compound')
                elif option == "Negative (Top 5)":
                    comments_to_display_df = temp_display_df[temp_display_df['sentiment_compound'] < 0].nsmallest(5, 'sentiment_compound')
                elif option == "Neutral (Top 5)":
                    neutral_comments = temp_display_df[temp_display_df['sentiment_compound'] == 0]
                    comments_to_display_df = neutral_comments.head(5) # just first 5 neutral
                else: # "All (Recent 5)" - since data is sorted by published_at ascending, tail(5) gets most recent
                    comments_to_display_df = temp_display_df.tail(5)


                if not comments_to_display_df.empty:
                    for i, (_, row) in enumerate(comments_to_display_df.iterrows()):
                        comment_text = row['text']
                        is_toxic_comment = row.get('is_toxic', False) # Use .get for safety
                        toxic_tag = " (Toxic)" if is_toxic_comment else ""
                        st.write(f"{i+1}. {comment_text}{toxic_tag}")
                else:
                    st.write("No comments to display for the selected filter.")


                # Keyword search within comments
                keyword = st.text_input("Enter a keyword to search within comments:")
                if keyword:
                    st.write(f"Comments containing the keyword '{keyword}':")
                    keyword_comments = data[data['text'].str.contains(keyword, case=False)]
                    for i, comment in enumerate(keyword_comments['text']):
                        st.write(f"{i+1}. {comment}")

                # Generate word cloud
                all_comments = ' '.join(data['cleaned_text'])
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)

                # Plot word cloud
                st.write("Word Cloud:")
                # Create a new figure for word cloud to avoid conflicts with timeline plot
                fig_wordcloud, ax_wordcloud = plt.subplots(figsize=(10,5))
                ax_wordcloud.imshow(wordcloud, interpolation='bilinear')
                ax_wordcloud.axis('off')
                st.pyplot(fig_wordcloud)

                # --- Sentiment Timeline and Emotion Drift Section ---
                st.markdown("## Sentiment Timeline and Emotion Drift Analysis")
                if 'sentiment_sliding_avg' in data.columns and 'is_drift_point' in data.columns:
                    # Generate the sentiment timeline plot
                    fig_timeline = plot_sentiment_timeline(data)
                    if fig_timeline:
                        st.pyplot(fig_timeline)
                    else:
                        st.info("Could not generate sentiment timeline plot (possibly due to missing data).")
                else:
                    st.info("Sentiment timeline data not available (possibly due to errors in earlier steps or insufficient data).")

                # --- Toxicity & Hate Speech Analysis Summary ---
                st.markdown("## Toxicity & Hate Speech Analysis")
                if 'is_toxic' in data.columns:
                    num_toxic_comments = data['is_toxic'].sum()
                    total_comments = len(data)
                    percentage_toxic = (num_toxic_comments / total_comments * 100) if total_comments > 0 else 0

                    st.metric("Total Comments Analyzed", total_comments)
                    st.metric("Comments Flagged as Toxic", num_toxic_comments)
                    st.metric("Percentage of Toxic Comments", f"{percentage_toxic:.2f}%")

                    if num_toxic_comments > 0:
                        st.markdown("#### Examples of Comments Flagged as Toxic:")
                        toxic_examples_df = data[data['is_toxic']].head(3) # Show up to 3 examples
                        for _, row_example in toxic_examples_df.iterrows():
                            # Display with caution. Maybe truncate or use expander.
                            with st.expander(f"Toxic Comment (Score: {row_example.get('toxicity_score', 0.0):.2f}) - Click to see"):
                                st.write(f"> {row_example['text']}")
                else:
                    st.info("Toxicity analysis data not available.")


                # --- RAG Query Engine Section ---
                st.markdown("## Query Comments with AI")

                # Get LLM and Embeddings model (cached)
                llm = get_llm()
                hf_embeddings = get_hf_embeddings()

                if llm and hf_embeddings: # Check if LLM (and thus API key) and embeddings are available
                    if 'cleaned_text' in data.columns and not data['cleaned_text'].empty:
                        # Pass hf_embeddings as a parameter to satisfy cache for create_comment_vector_store
                        comment_retriever = create_comment_vector_store(data, _embeddings_model=hf_embeddings)

                        if comment_retriever:
                            # Pass comment_retriever and llm to satisfy cache for setup_rag_chain
                            rag_chain = setup_rag_chain(_retriever=comment_retriever, _llm=llm)

                            if rag_chain:
                                user_question = st.text_input("Ask a question about the comments:")
                                if user_question:
                                    with st.spinner("Searching for answers in comments..."):
                                        try:
                                            # The chain expects a dictionary with a 'query' key
                                            answer = rag_chain.invoke({'query': user_question})
                                            st.markdown("### Answer:")
                                            st.write(answer.get('result', "Sorry, I couldn't find an answer."))
                                            # if answer.get('source_documents'):
                                            #     st.markdown("#### Sources:")
                                            #     for doc in answer['source_documents']:
                                            #         st.write(doc.page_content) # Displaying content of source docs
                                        except Exception as e:
                                            st.error(f"Error querying RAG chain: {e}")
                            else:
                                st.info("AI Query Engine could not be initialized.")
                        else:
                            st.info("Could not build knowledge base from comments for AI queries.")
                    else:
                        st.info("Not enough comment data to build AI knowledge base.")
                else:
                    st.info("AI Query Engine disabled due to missing OpenAI API Key or embedding model initialization failure.")

            else: # This 'else' corresponds to 'if data is not None and not data.empty:'
                  # If 'data' is None or empty after graph execution (e.g. fetch error)
                if not final_state.get("error_message"): # If no specific error message from graph
                    st.warning("No comment data processed by the workflow to display.")
                # Error message from graph already displayed if it exists.
                pass # No further processing if data is None

                # --- Comment Cluster Analysis Section ---
                st.markdown("## Comment Cluster Analysis")
                if data is not None and 'cluster_label' in data.columns and 'tsne_x' in data.columns and 'tsne_y' in data.columns:
                    if data['cluster_label'].nunique() > 0 and not data[['tsne_x', 'tsne_y']].isnull().all().all():
                        st.write("### Comment Clusters (t-SNE Visualization)")

                        fig_cluster, ax_cluster = plt.subplots(figsize=(10, 8))
                        # Use a color map that provides distinct colors for a reasonable number of clusters
                        # Filter out points where t-SNE results might be NaN if any partial failures occurred
                        plot_df = data.dropna(subset=['tsne_x', 'tsne_y', 'cluster_label'])
                        # Ensure cluster labels are integers for coloring if they aren't already
                        plot_df['cluster_label'] = plot_df['cluster_label'].astype(int)

                        unique_labels = sorted(plot_df['cluster_label'].unique())
                        # Filter out -1 if it means "not clustered"
                        unique_labels = [l for l in unique_labels if l != -1]

                        # Create a color map based on the number of unique, valid cluster labels
                        colors = plt.cm.get_cmap('viridis', len(unique_labels)) if len(unique_labels) > 0 else None

                        if colors:
                            for i, cluster_id in enumerate(unique_labels):
                                cluster_data = plot_df[plot_df['cluster_label'] == cluster_id]
                                ax_cluster.scatter(cluster_data['tsne_x'], cluster_data['tsne_y'], label=f'Cluster {cluster_id}', color=colors(i), alpha=0.7, s=50)

                            ax_cluster.set_title('t-SNE Visualization of Comment Clusters')
                            ax_cluster.set_xlabel('t-SNE Dimension 1')
                            ax_cluster.set_ylabel('t-SNE Dimension 2')
                            if len(unique_labels) > 1 : # Only show legend if multiple clusters
                                ax_cluster.legend()
                            ax_cluster.grid(True)
                            st.pyplot(fig_cluster)
                        else:
                            st.info("Not enough distinct clusters to plot or t-SNE data is invalid.")

                        # Display LLM Cluster Summaries
                        cluster_summaries = final_state.get("cluster_summaries")
                        if cluster_summaries:
                            st.write("### Cluster Theme Summaries (AI Generated)")
                            for cluster_id, summary in sorted(cluster_summaries.items()):
                                st.markdown(f"**Cluster {cluster_id} Theme:** {summary}")
                        else:
                            st.info("No AI-generated cluster summaries available.")
                    else:
                        st.info("Not enough data or clusters for t-SNE visualization or no valid cluster labels assigned.")
                else:
                    st.info("Clustering data not available (possibly due to errors in earlier steps or insufficient data).")

                # --- LangFlow Informational Section ---
                with st.expander("Advanced Workflow Design with LangFlow", expanded=False):
                    st.markdown("""
                        ### Advanced Workflow Design with LangFlow

                        For users interested in visually designing more complex AI workflows or experimenting with different LangChain components and agentic structures, we recommend exploring **LangFlow**.

                        LangFlow is a graphical user interface (GUI) for LangChain that allows you to drag-and-drop components, connect them, and build intricate flows.

                        **How to use it with this application:**

                        1.  **Prototype in LangFlow:** Use the standalone LangFlow application (available at [https://langflow.org/](https://langflow.org/) or by running it locally) to design and test your desired workflows.
                        2.  **Gain Insights:** Understand how different components (LLMs, retrievers, tools) can work together.
                        3.  **Apply Concepts:** While direct import of LangFlow designs into this application is not currently supported, you can use the insights gained from LangFlow to better utilize the features available here (like the RAG query engine or the automated analysis) or manually adapt simplified versions of your LangFlow creations if you are developing custom Python scripts.

                        This approach allows you to leverage the power of visual prototyping in LangFlow while using this application for streamlined YouTube comment analysis.
                    """)

if __name__ == "__main__":
    main()
