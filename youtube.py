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
import bs4
import urllib.request

# Global variables
token_file = "fb_token.json"
model_file = "response_model.pkl"
pygame.mixer.init()
history = deque(maxlen=10)  # Store last 10 interactions

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

def get_google_search_result(query):
    """Search Google and get the first relevant result's content."""
    try:
        search_results = list(search(query, num_results=3))
        
        for url in search_results:
            try:
                html = urllib.request.urlopen(url).read()
                soup = bs4.BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                sentences = text.split('.')[:3]
                answer = '. '.join(sentences) + '.'
                
                if len(answer) > 50:
                    return answer
                    
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                continue
                
        return "I couldn't find a clear answer to that question."
        
    except Exception as e:
        print(f"Google search error: {e}")
        return "I'm having trouble searching for an answer right now."

def speak(text):
    """Speak text using gTTS."""
    global last_spoken_time
    current_time = time.time()
    
    if current_time - last_spoken_time >= SPEAK_DELAY and text and text.strip():
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
                # speak(f"Now playing {video_title}")

                # Play the audio without sleep
                pygame.mixer.music.load(mp3_file)
                pygame.mixer.music.set_volume(0.8)
                pygame.mixer.music.play()

                # Wait until playback ends
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                # Cleanup will only happen after the music stops
                pygame.mixer.music.unload()
                
            except Exception as e:
                print(f"Download/playback error: {e}")
                speak("Sorry, I encountered an error with the song")

    except Exception as e:
        print(f"Fatal error in play_youtube_song: {e}")
        speak("Sorry, I'm having trouble playing music right now")

def handle_question(text):
    """Enhanced question handling with Wikipedia and Google search fallback."""
    try:
        search_text = text.lower()
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'is', 'are', 'can', 'could', 'tell', 'me', 'about']
        for word in question_words:
            search_text = search_text.replace(word, '')
        search_text = search_text.strip()
        
        if not search_text:
            speak("I'm not sure what you're asking about. Could you rephrase your question?")
            return
            
        # Try Wikipedia first
        try:
            summary = wikipedia.summary(search_text, sentences=2)
            speak(summary)
            return
            
        except (wikipedia.exceptions.DisambiguationError, 
                wikipedia.exceptions.PageError, 
                Exception) as e:
            print(f"Wikipedia error: {e}, falling back to Google search")
            
            # Fallback to Google search
            answer = get_google_search_result(text)
            speak(answer)
            
    except Exception as e:
        print(f"Error handling question: {e}")
        speak("I'm sorry, I had trouble finding an answer to your question.")

def send_message(platform, recipient, message):
    """Send message via specified platform."""
    speak(f"Sending message via {platform} to {recipient}: {message}")
    # Implement actual message sending logic here

def continuous_recording():
    """Continuously listen for voice input and process commands."""
    try:
        with microphone as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                print("Processing speech...")
                
                try:
                    text = recognizer.recognize_google(audio)
                    print(f"You said: {text}")
                    
                    # Check for exit commands
                    if text.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                        speak("Goodbye!")
                        return False
                        
                    return text
                    
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    speak("I didn't catch that. Could you please repeat?")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    speak("I'm having trouble with speech recognition.")
                    
            except sr.WaitTimeoutError:
                print("No speech detected within timeout")
                return None
    except Exception as e:
        print(f"Error in continuous recording: {e}")
        speak("Sorry, there was an error in continuous recording")
    
    return None

def main():
    """Main function to run the assistant."""
    global vectorizer, model
    vectorizer, model = train_prediction_model()
    
    running = True
    while running:
        # Continuously listen for input
        user_text = continuous_recording()
        
        if user_text is False:
            running = False
            break
        elif user_text:
            history.append(user_text)
            action = predict_action(user_text, vectorizer, model)
            
            if action == 'music':
                song_query = user_text.replace("play", "").strip()
                play_youtube_song(song_query)
                
            elif action == 'question':
                handle_question(user_text)
                
            elif action == 'message':
                # Here, you'll need further context to determine recipient and platform
                recipient = "someone"  # Placeholder
                send_message("WhatsApp", recipient, user_text)
                
            elif action == 'exit':
                speak("Goodbye!")
                running = False

if __name__ == "__main__":
    main()
