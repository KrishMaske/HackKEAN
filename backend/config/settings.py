import os
import dotenv
from dotenv import load_dotenv
from google import genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
google_client= genai.Client(api_key=GOOGLE_API_KEY)