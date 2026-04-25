import requests

API_URL = "https://api-inference.huggingface.co/models/IDEA-Research/grounding-dino-base"
headers = {} # No token to see if it works for free

try:
    print("Testing Hugging Face API for Grounding DINO...")
    # Send bus image and prompt "bus"
    with open("assets/uploads/STRANGER_THINGS_CLIP.mp4", "rb") as f:
        # just need an image. I'll download bus.png
        pass

    import urllib.request
    urllib.request.urlretrieve("https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png", "bus.png")
    
    with open("bus.png", "rb") as f:
        data = f.read()
        
    response = requests.post(API_URL, headers=headers, data=data)
    print("STATUS:", response.status_code)
    print(response.json())
except Exception as e:
    print("ERROR:", e)
