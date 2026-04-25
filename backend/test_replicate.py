import replicate
import os
from dotenv import load_dotenv

load_dotenv()

try:
    print("Testing Replicate segment-anything API...")
    output = replicate.run(
        "meta/segment-anything:dfaf74e304899c75953594b29332e2978d3eb6cefa771bd0b55edeb21307b22d",
        input={
            "image": "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
            "mask_limit": 1
        }
    )
    print("SUCCESS!")
    print(output)
except Exception as e:
    print("ERROR:", e)
