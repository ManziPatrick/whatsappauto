import time
import json
import os
import requests
from googlesearch import search
import wikipedia
from gtts import gTTS
import pygame
import yt_dlp
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pickle
import threading
from collections import deque
import webbrowser
import bs4

token_file = "fb_token.json"
model_file = "response_model.pkl"
pygame.mixer.init()
history = deque(maxlen=10)  

# Initialize speech recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()

last_spoken_time = 0
SPEAK_DELAY = 2  # Minimum seconds between voice outputs

# Training data for basic prediction
initial_training_data = {
    'inputs': [
        "play music",
        "play song",
        "search for",
        "tell me about",
        "send message",
        "send whatsapp",
        "what is",
        "how to",
        "goodbye",
        "exit"
    ],
    'responses': [
        "music",
        "music",
        "question",
        "question",
        "message",
        "message",
        "question",
        "question",
        "exit",
        "exit"
    ]
}

def train_prediction_model():
    """Train a simple prediction model for responses."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(initial_training_data['inputs'])
    y = initial_training_data['responses']
    
    model = MultinomialNB()
    model.fit(X, y)
    
    # Save the model and vectorizer
    with open(model_file, 'wb') as f:
        pickle.dump((vectorizer, model), f)
    
    return vectorizer, model

def predict_action(text, vectorizer, model):
    """Predict the most likely action based on input text."""
    try:
        X = vectorizer.transform([text.lower()])
        prediction = model.predict(X)[0]
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return None
# ... (previous imports remain the same)

def get_search_labs_info(query):
    """Search using Google Search Labs and provide a summarized response."""
    try:
        # Modify query to focus on authoritative sources
        modified_query = f"{query} site:.edu OR site:.gov OR site:.org"
        
        # Perform Google search with correct parameters
        # The search() function uses 'num' instead of 'num_results'
        search_results = list(search(modified_query, num=3, stop=3))
        
        if not search_results:
            return "I couldn't find clear information from authoritative sources."

        combined_content = []
        
        # Try to get content from top 3 results
        for result_url in search_results:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(result_url, timeout=5, headers=headers)
                soup = bs4.BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                
                # Extract main content
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                if main_content:
                    # Get paragraphs
                    paragraphs = main_content.find_all('p')
                    content = ' '.join(p.get_text().strip() for p in paragraphs)
                    # Basic cleaning of the content
                    content = ' '.join(content.split())  # Remove extra whitespace
                    if content:  # Only add non-empty content
                        combined_content.append(content)
            except Exception as e:
                print(f"Error processing URL {result_url}: {e}")
                continue
        
        if not combined_content:
            return "I found some sources but couldn't extract meaningful content."
        
        # Combine and summarize content
        all_content = ' '.join(combined_content)
        
        # Basic summarization - take first few sentences
        sentences = [s.strip() for s in all_content.split('.') if s.strip()]
        summary = '. '.join(sentences[:3]) + '.'
        
        return summary

    except Exception as e:
        print(f"Search Labs error: {e}")
        return None

# The rest of the code remains the same...
def speak(text):
    """Speak text using gTTS and disable the microphone during speech."""
    global last_spoken_time
    current_time = time.time()
    
    if current_time - last_spoken_time >= SPEAK_DELAY and text and text.strip():
        # Disable the microphone
        print("Microphone disabled.")

        def speak_thread():
            try:
                tts = gTTS(text, lang='en', slow=False)
                temp_file = f"speech_{int(time.time())}.mp3"
                tts.save(temp_file)
                
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
                pygame.mixer.music.unload()
                os.remove(temp_file)

            except Exception as e:
                print(f"Speech error: {e}")

        thread = threading.Thread(target=speak_thread)
        thread.start()
        last_spoken_time = current_time
        thread.join()  # Wait for the speech to finish

def listen_for_speech():
    """Listen for speech input and convert to text."""
    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("Processing speech...")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

def play_youtube_song(song_query):
    """Search for and play audio from a YouTube video matching the query."""
    try:
        # Configure yt-dlp options for audio only
        ydl_opts = {
            'format': 'bestaudio',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'noplaylist': True,
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'prefer_ffmpeg': True,
            'outtmpl': 'temp_audio_%(id)s.%(ext)s'
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Search for the video first
            speak("Searching for your song...")
            try:
                info = ydl.extract_info(f"ytsearch1:{song_query}", download=True)
                if not info or 'entries' not in info or not info['entries']:
                    speak("Sorry, I couldn't find that song")
                    return

                video = info['entries'][0]
                video_title = video.get('title', 'Unknown')
                mp3_file = f"temp_audio_{video['id']}.mp3"
                
                # Announce the title before loading the music file
                speak(f"Now playing {video_title}")

                # Play the audio
                pygame.mixer.music.load(mp3_file)
                pygame.mixer.music.set_volume(0.8)
                pygame.mixer.music.play()

                # Wait until playback ends
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                # Cleanup will only happen after the music stops
                pygame.mixer.music.unload()
                os.remove(mp3_file)  # Clean up the temporary file
                
            except Exception as e:
                print(f"Download/playback error: {e}")
                speak("Sorry, I encountered an error with the song")

    except Exception as e:
        print(f"Fatal error in play_youtube_song: {e}")
        speak("Sorry, I'm having trouble playing music right now")

def handle_question(text):
    """Enhanced question handling with Google search first, then Wikipedia fallback."""
    try:
        # Clean up the search text
        search_text = text.lower()
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'is', 'are', 'can', 'could', 'tell', 'me', 'about']
        
        for word in question_words:
            search_text = search_text.replace(word, '')
        
        search_text = search_text.strip()
        
        if not search_text:
            speak("I'm not sure what you're asking about. Could you rephrase your question?")
            return
        
        # First try Google Search Labs
        google_response = get_search_labs_info(search_text)
        
        if google_response:
            speak("Based on my search:")
            speak(google_response)
            
            # Try Wikipedia for additional context
            try:
                wiki_summary = wikipedia.summary(search_text, sentences=2)
                speak("Additional information from Wikipedia:")
                speak(wiki_summary)
            except Exception as wiki_error:
                print(f"Wikipedia fallback error: {wiki_error}")
                # Don't speak about the Wikipedia error since we already have Google results
        else:
            # If Google search fails, try Wikipedia as fallback
            try:
                wiki_summary = wikipedia.summary(search_text, sentences=3)
                speak("Here's what I found from Wikipedia:")
                speak(wiki_summary)
            except Exception as wiki_error:
                print(f"Wikipedia error: {wiki_error}")
                speak("I'm sorry, I couldn't find reliable information about that topic.")
                
    except Exception as e:
        print(f"Error handling question: {e}")
        speak("I'm sorry, I had trouble finding an answer to your question.")

def send_message(platform, recipient, message):
    """Placeholder for sending messages."""
    speak(f"Sending message to {recipient} on {platform}: {message}")
    # Implement actual message sending logic here

# Load or train the prediction model on startup
if os.path.exists(model_file):
    with open(model_file, 'rb') as f:
        vectorizer, model = pickle.load(f)
else:
    vectorizer, model = train_prediction_model()

def continuous_recording():
    """Continuously listen for speech and process actions."""
    while True:
        try:
            text = listen_for_speech()
            if text:
                # Store the last spoken text for history
                history.append(text)
                
                # Check for exit commands
                if text.lower() in ['exit', 'quit', 'goodbye']:
                    speak("Goodbye!")
                    break
                
                # Predict action
                action = predict_action(text, vectorizer, model)
                
                if action == "music":
                    play_youtube_song(text)
                elif action == "message":
                    # Extract recipient and message (this is a placeholder)
                    send_message("WhatsApp", "John Doe", "Hello!")
                elif action == "exit":
                    speak("Goodbye!")
                    break
                else:
                    handle_question(text)
                    
        except Exception as e:
            print(f"Error in continuous recording: {e}")

def main():
    """Main function to run the voice assistant."""
    try:
        speak("Hello! I'm your voice assistant. How can I help you?")
        
        # Start the continuous recording in a separate thread
        recording_thread = threading.Thread(target=continuous_recording, daemon=True)
        recording_thread.start()
        
        # Keep the main thread alive
        while recording_thread.is_alive():
            time.sleep(1)
            
    except KeyboardInterrupt:
        speak("Goodbye!")
        print("Exiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        speak("I encountered an error and need to shut down.")

if __name__ == "__main__":
    main()