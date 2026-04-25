import requests
import base64

API_URL = "https://api-inference.huggingface.co/models/facebook/sam-vit-base"
headers = {} # Try without token

try:
    print("Testing Hugging Face SAM API...")
    with open("bus.png", "rb") as f:
        data = f.read()
    
    # Hugging Face image segmentation API usually doesn't support bounding boxes via simple inference API, 
    # it only supports zero-shot (returns all masks) or you have to use a specific pipeline.
    # Let's try sending just the image data to see if it returns any masks.
    response = requests.post(API_URL, headers=headers, data=data)
    print("STATUS:", response.status_code)
    # print(response.json()[:1]) # don't print huge payload
except Exception as e:
    print("ERROR:", e)
