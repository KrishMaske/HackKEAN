import time
import os
from gradio_client import Client, handle_file

try:
    print("Testing Grounded-SAM API...")
    client = Client("IDEA-Research/Grounded-Segment-Anything")
    result = client.predict(
        image_in=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
        text_prompt_in="bus",
        task_type="text to mask",
        api_name="/run_grounded_sam"
    )
    print("SUCCESS!")
    print(result)
except Exception as e:
    print("ERROR:", e)
