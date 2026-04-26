import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "hello"}],
    )
    print("Text model success:", response.choices[0].message.content)
except Exception as e:
    print("Text model error:", e)
