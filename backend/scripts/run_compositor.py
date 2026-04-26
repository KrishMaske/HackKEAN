import cv2
import numpy as np
import os

def main():
    source_vid = "assets/input/FastAndFurious.mp4"
    mask_vid = "assets/masks/fast_and_furious_mask.mp4"
    miata_img_path = "assets/pink_miata_transparent.png"
    output_vid = "assets/output/fast_and_furious_final.mp4"

    os.makedirs(os.path.dirname(output_vid), exist_ok=True)

    cap_src = cv2.VideoCapture(source_vid)
    cap_mask = cv2.VideoCapture(mask_vid)

    fps = cap_src.get(cv2.CAP_PROP_FPS)
    width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_vid, fourcc, fps, (width, height))

    miata = cv2.imread(miata_img_path, cv2.IMREAD_UNCHANGED)
    if miata is None:
        print(f"Error: Could not load {miata_img_path}")
        return

    # Ensure miata has 4 channels
    if miata.shape[2] == 3:
        miata = cv2.cvtColor(miata, cv2.COLOR_BGR2BGRA)

    # Flip miata horizontally to match the car's direction in the scene
    miata = cv2.flip(miata, 1)

    miata_rgb = miata[:, :, :3]
    miata_alpha = miata[:, :, 3]

    frame_count = 0
    while True:
        ret_src, frame = cap_src.read()
        ret_mask, mask_frame = cap_mask.read()

        if not ret_src or not ret_mask:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Compositing frame {frame_count}/{total_frames}...")

        # Convert mask to single channel binary
        gray_mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

        # Instead of using the dynamic mask bounding box which squishes the Miata and jumps around
        # during occlusion, we use the stable bounding box from the JSON metadata.
        x, y, w, h = 200, 230, 850, 370

        if w > 0 and h > 0:
            # Resize miata to fit the bounding box exactly
            resized_miata = cv2.resize(miata, (w, h))
            res_miata_rgb = resized_miata[:, :, :3]
            res_miata_alpha = resized_miata[:, :, 3]

            # ── Lighting Matching ──
            # Calculate average Value (brightness) of original car inside the mask
            roi_frame = frame[y:y+h, x:x+w]
            roi_mask = binary_mask[y:y+h, x:x+w]
            
            hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
            # Mean V where mask is active
            if cv2.countNonZero(roi_mask) > 0:
                mean_v_orig = cv2.mean(hsv_roi[:, :, 2], mask=roi_mask)[0]
                
                # Calculate mean V of resized miata
                hsv_miata = cv2.cvtColor(res_miata_rgb, cv2.COLOR_BGR2HSV)
                mean_v_miata = cv2.mean(hsv_miata[:, :, 2], mask=res_miata_alpha)[0]
                
                if mean_v_miata > 0:
                    scale = mean_v_orig / mean_v_miata
                    # Apply scale, bounded to 255
                    hsv_miata[:, :, 2] = np.clip(hsv_miata[:, :, 2] * scale, 0, 255).astype(np.uint8)
                    res_miata_rgb = cv2.cvtColor(hsv_miata, cv2.COLOR_HSV2BGR)

            # ── Compositing ──
            # Create a combined alpha mask for the ROI
            alpha_f = res_miata_alpha.astype(float) / 255.0
            mask_f = roi_mask.astype(float) / 255.0
            
            # The final alpha is where BOTH the Miata has opacity AND the original mask is active.
            final_alpha = alpha_f * mask_f
            
            # Expand final_alpha to 3 channels for broadcasting
            final_alpha_3 = np.dstack([final_alpha, final_alpha, final_alpha])
            
            # Blend
            frame_roi = frame[y:y+h, x:x+w]
            blended = (res_miata_rgb * final_alpha_3 + frame_roi * (1 - final_alpha_3)).astype(np.uint8)
            
            frame[y:y+h, x:x+w] = blended

        out.write(frame)

    cap_src.release()
    cap_mask.release()
    out.release()
    print(f"Done! Saved to {output_vid}")

if __name__ == "__main__":
    main()
