import os
import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    with open("temp_test_frame.jpg", "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
        
    response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
    )
    print("Vision model success:", response.choices[0].message.content)
except Exception as e:
    print("Vision model error:", e)
