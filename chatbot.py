import streamlit as st
import nltk
import spotipy
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import googleapiclient.discovery
from spotipy.oauth2 import SpotifyClientCredentials
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

# Initialize NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize Spotipy
client_id = '#########################'
client_secret = '######################'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Initialize CSV
csv_filename = "user_details.csv"

def initialize_csv():
    try:
        df = pd.DataFrame(columns=['Name', 'Song', 'YouTube Link'])
        df.to_csv(csv_filename, index=False)
    except Exception as e:
        st.error("Error initializing CSV:", e)

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
            video_id = search_results[0]['href'][9:]
            return video_id
        except Exception as e:
            st.error("Error fetching video ID:", e)
            return None

#Scrap the comments for video
def fetch_comments_with_api(video_id, max_comments=1500):
    try:
        api_service_name = "youtube"
        api_version = "v3"
        DEVELOPER_KEY = "################################" 

        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey=DEVELOPER_KEY)

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

# Function to recommend songs based on popularity
def recommend_songs_based_on_popularity(song_input):
    results = sp.search(q=song_input, limit=1, type='track')
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        artist = track['artists'][0]['name']
        song_name = track['name']
        artist_uri = track['artists'][0]['uri']
        top_tracks = sp.artist_top_tracks(artist_uri)
        recommended_tracks = []
        for idx, track in enumerate(top_tracks['tracks']):
            recommended_tracks.append((track['name'], track['popularity']))
            if idx == 9:  
                break
        return recommended_tracks
    else:
        return None

# Plot waveform
def plot_waveform(song_id):
    # Retrieve audio analysis for the song
    audio_analysis = sp.audio_analysis(song_id)

    # Extract waveform data
    segments = audio_analysis['segments']
    loudness_max = [segment['loudness_max'] for segment in segments]
    start = [segment['start'] for segment in segments]

    # Create waveform plot using scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=start, y=loudness_max, mode='lines', name='Waveform'))
    fig.update_layout(title='Waveform Graph', xaxis_title='Time (s)', yaxis_title='Loudness (dB)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# Plot spider graph
def plot_spider_graph(song_features):
    attributes = list(song_features.keys())
    values = list(song_features.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
          r=values,
          theta=attributes,
          fill='toself',
          name='Song Features',
          line=dict(color='green')
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1] # Set the range for the radial axis
        )),
      showlegend=True,
      title='Radar Chart for Song Features',
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# Streamlit App
def main():
    st.image("E:/Song Chatbot/top_image.jpg")  
    st.title("Song Chatbot!")
    name = st.text_input("Enter your name:")
    
    if name:
        st.write(f"Hello, {name}!")

        # Ask the user if they're in the mood for music
        mood_response = st.radio("Are you in the mood for some music?", ("Yes", "No"))

        if mood_response == "Yes":
            st.write("Great! Please enter the name of the song you'd like to know more about.")
            # Ask the user to enter a song
            user_input = st.text_input("Enter name of the song:")

            if user_input:
                # If the user has entered a song name, search for it on Spotify
                results = sp.search(q=user_input, limit=1, type='track')

                if results['tracks']['items']:
                    # If the song is found, display its details
                    track = results['tracks']['items'][0]
                    song_name = track['name']
                    artist = track['artists'][0]['name']
                    year = track['album']['release_date'][:4]
                    popularity = track['popularity']
                    album_name = track['album']['name'] if 'album' in track else None

                    st.write(f"*Song Name:* {song_name}")
                    st.write(f"*Artist:* {artist}")
                    if album_name:
                        st.write(f"*Album:* {album_name}")
                    st.write(f"*Year:* {year}")
                    st.write(f"*Popularity:* {popularity}")

                    # Ask the user for the YouTube link
                    youtube_link = st.text_input("Please enter the YouTube link for the song:")

                    if youtube_link:
                        st.write(f"Thank you for providing the YouTube link: {youtube_link}")

                        # Fetch comments for the song
                        data = fetch_comments(youtube_link, max_comments=1500)  
                        if data is not None:
                            st.write("Top 5 Comments:")
                            cleaned_comments = data['text'].apply(clean_and_analyze_text)
                            cleaned_comments_text, sentiment_scores = zip(*cleaned_comments)
                            data['cleaned_text'] = cleaned_comments_text
                            data['sentiment_scores'] = sentiment_scores

                            option = st.radio("Select comments to display:", ["All", "Positive", "Negative", "Neutral"], index=0)

                            if option == "Positive":
                                top_comments = data[data['sentiment_scores'].apply(lambda x: x['compound']) > 0]['text'].head(5)
                            elif option == "Negative":
                                top_comments = data[data['sentiment_scores'].apply(lambda x: x['compound']) < 0]['text'].head(5)
                            elif option == "Neutral":
                                top_comments = data[data['sentiment_scores'].apply(lambda x: x['compound']) == 0]['text'].head(5)
                            else:
                                top_comments = data['text'].head(5)

                            for i, comment in enumerate(top_comments):
                                st.write(f"{i+1}. {comment}")

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
                            plt.figure(figsize=(10, 5))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot(plt)

                            # Recommend songs based on popularity
                            recommended_songs = recommend_songs_based_on_popularity(song_name)
                            if recommended_songs:
                                st.write("Top 10 Recommended Songs Based on Popularity:")
                                for idx, (song, popularity) in enumerate(recommended_songs, start=1):
                                    st.write(f"{idx}. {song} (Popularity: {popularity})")
                            else:
                                st.warning("Couldn't find recommendations for similar songs.")

                            # Store user details, song name, and link in CSV
                            try:
                                df = pd.DataFrame([[name, song_name, youtube_link]], columns=['Name', 'Song', 'YouTube Link'])
                                df.to_csv(csv_filename, mode='a', header=False, index=False)
                                st.success("Details stored successfully!")
                            except Exception as e:
                                st.error("Error storing details:", e)
                            
                            # Plot waveform, heatmap, and spider graph
                            audio_features = sp.audio_features(results['tracks']['items'][0]['id'])[0]
                            st.markdown('## Song Analysis Feature Analysis')
                            st.plotly_chart(plot_waveform(results['tracks']['items'][0]['id']), use_container_width=True)

                            st.plotly_chart(plot_spider_graph(audio_features), use_container_width=True)
                        else:
                            st.error("Failed to fetch comments.")
                else:
                    st.warning("Sorry, I couldn't find information about that song. Please try another one.")
        else:
            st.write("No problem! Feel free to reach out if you change your mind.")

if __name__ == "__main__":
    initialize_csv()
    main()
