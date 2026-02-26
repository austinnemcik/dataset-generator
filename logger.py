from datetime import datetime
import os
LOG_FILE = "logs/log.txt"
def saveToLog(message: str):
    now = datetime.now()
    formatted_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    with open (LOG_FILE, "a", encoding="utf-8") as file:
        file.write(f"{formatted_timestamp}: {message}")
        return True

        # later we can make log location from .env so it's a little more flexible but need default anyways.

def deleteLog():
    try: 
        os.remove(LOG_FILE)
        print(f"Successfully removed file at {LOG_FILE}")
    except OSError as e:
        print(f"Error: {e.strerror}")
    