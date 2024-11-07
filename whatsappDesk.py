import subprocess
import pyautogui
import time
from urllib.parse import quote
from contacts_manager import search_contact, remove_words, speak


def whatsApp(mobile_no, message, action, name):
    target_tab = {'message': 12, 'call': 7, 'video': 6}[action]
    jarvis_message = f"{action.capitalize()} {'sent' if action == 'message' else 'initiated'} successfully to {name}"

    encoded_message = quote(message) if action == 'message' else ''
    whatsapp_url = f"whatsapp://send?phone={mobile_no}&text={encoded_message}"

    subprocess.run(f'start "" "{whatsapp_url}"', shell=True)
    time.sleep(5)

    for _ in range(target_tab):
        pyautogui.hotkey('tab')

    pyautogui.hotkey('enter')
    speak(jarvis_message)

def find_and_send_whatsapp(query):
    words_to_remove = ['make', 'a', 'to', 'phone', 'call', 'send', 'message', 'whatsapp', 'video']
    filtered_query = remove_words(query, words_to_remove)

    contact_no = search_contact(filtered_query)
    if contact_no:
        if not contact_no.startswith('+25'):
            contact_no = '+25' + contact_no 

        action = "message" if "send message" in query else "call" if "phone call" in query else "video"
        message_content = ""

        if action == "message":
            speak("What message would you like to send?")
            message_content = take_command()
        
        whatsApp(contact_no, message_content, action, filtered_query)
    else:
        speak("Contact not found.")
