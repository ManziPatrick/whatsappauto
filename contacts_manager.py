import sqlite3
import csv
from gtts import gTTS
import speech_recognition as sr

# Connect to the database
con = sqlite3.connect('contacts.db')
cursor = con.cursor()

# Create the contacts table if it doesn't already exist
cursor.execute('''CREATE TABLE IF NOT EXISTS contacts (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(200),
                    mobile_no VARCHAR(255),
                    email VARCHAR(255) NULL)''')
con.commit()

def import_contacts_from_csv(file_path, column_indices=[0, 30]):
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  
        for row in csvreader:
            selected_data = [row[i] for i in column_indices]
            cursor.execute('''INSERT INTO contacts (id, name, mobile_no) VALUES (null, ?, ?);''', tuple(selected_data))
    con.commit()
    print("Contacts imported successfully.")

def remove_words(input_string, words_to_remove):
    words = input_string.split()
    filtered_words = [word for word in words if word.lower() not in words_to_remove]
    return ' '.join(filtered_words)

def search_contact(query):
    query = query.strip().lower()
    cursor.execute("SELECT mobile_no FROM contacts WHERE LOWER(name) LIKE ? OR LOWER(name) LIKE ?", ('%' + query + '%', query + '%'))
    results = cursor.fetchall()
    return results[0][0] if results else None

# Function to handle speaking
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("speech.mp3")
    os.system("mpg321 speech.mp3")  # or any command to play mp3, depending on OS

# Voice command recognition
def take_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"User said: {query}")
    except sr.UnknownValueError:
        print("Sorry, I did not understand.")
        return ""
    return query.lower()
