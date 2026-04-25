from pydantic_settings import BaseSettings
from google import genai

class Settings(BaseSettings):
    google_api_key: str
    mongodb_uri: str

    @property
    def google_client(self):
        """
        Dynamically initializes the Google GenAI client using the API key.
        """
        return genai.Client(api_key=self.google_api_key)

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()