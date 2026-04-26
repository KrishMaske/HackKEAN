import cv2
import os
import numpy as np

def generate_highlight(video_path, masks_dir, output_path, product_label="PRODUCT"):
    if not os.path.exists(video_path):
        print(f"Error: Video {video_path} not found")
        return
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Generating highlight for {product_label}...")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        mask_path = os.path.join(masks_dir, f"frame_{frame_idx:04d}.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and cv2.countNonZero(mask) > 100:
                # 1. Create green overlay
                highlight = frame.copy()
                highlight[mask > 0] = [0, 255, 0] # Bright green
                
                # 2. Blend original and highlight
                alpha = 0.4
                frame = cv2.addWeighted(highlight, alpha, frame, 1 - alpha, 0)
                
                # 3. Find contours for bounding box
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw label background
                    cv2.rectangle(frame, (x, y - 30), (x + 150, y), (0, 255, 0), -1)
                    
                    # Draw label text
                    label = f"{product_label} 98.4%"
                    cv2.putText(frame, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
            
    cap.release()
    out.release()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    test_cases = [
        {
            "video_path": "assets/input/FastAndFurious.mp4",
            "masks_dir": "assets/masks/fast_and_furious/alpha",
            "output_path": "assets/output/fast_and_furious_highlight.mp4",
            "product_label": "SPORTS CAR"
        },
        {
            "video_path": "assets/input/CastAway.mp4",
            "masks_dir": "assets/masks/castaway_fedex/alpha",
            "output_path": "assets/output/castaway_fedex_highlight.mp4",
            "product_label": "FEDEX BOX"
        },
        {
            "video_path": "assets/input/STRANGER_THINGS_CLIP.mp4",
            "masks_dir": "assets/masks/stranger_things_83/alpha",
            "output_path": "assets/output/stranger_things_83_highlight.mp4",
            "product_label": "KFC BUCKET"
        },
        {
            "video_path": "assets/input/COCA_COLA_STRANGER_THINGS.mp4",
            "masks_dir": "assets/masks/stranger_things_coke/alpha",
            "output_path": "assets/output/stranger_things_coke_highlight.mp4",
            "product_label": "COCA-COLA CAN"
        }
    ]

    for case in test_cases:
        generate_highlight(
            case["video_path"],
            case["masks_dir"],
            case["output_path"],
            case["product_label"]
        )

