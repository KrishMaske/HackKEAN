from pydantic_settings import BaseSettings
from google import genai
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    
    @property
    def google_client(self):
        """
        Dynamically initializes the Google GenAI client using the API key.
        """
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set")
        return genai.Client(api_key=self.google_api_key)

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
