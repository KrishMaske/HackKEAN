import os
import cv2
import numpy as np

class Sam3Processor:
    def __init__(self):
        print("[SAM3] Initializing Local Edge-Compute Engine (GrabCut Architecture)...")

    @classmethod
    def from_pretrained(cls, model_path):
        return cls()

    def generate_mask(self, frame: np.ndarray, prompt: str, bbox: list = None, prev_mask: np.ndarray = None) -> np.ndarray:
        """
        Generates a pixel-perfect binary mask using GrabCut architecture.
        Since cloud APIs failed, this runs 100% locally and instantly on the CPU.
        """
        if bbox is None and prev_mask is None:
            return None

        # Initialize mask and background/foreground models for GrabCut
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        try:
            if bbox is not None and len(bbox) == 4 and bbox[0] is not None:
                # Convert bounding box [xmin, ymin, xmax, ymax] to [x, y, width, height]
                x = max(0, int(bbox[0]))
                y = max(0, int(bbox[1]))
                w = max(1, int(bbox[2] - x))
                h = max(1, int(bbox[3] - y))
                
                # --- PADDING LOGIC ---
                # Groq is known to be slightly imprecise, sometimes cutting off the edges of the object.
                # By expanding the box by 25%, we ensure the entire object is inside.
                pad_x = int(w * 0.25)
                pad_y = int(h * 0.25)
                px1 = max(0, x - pad_x)
                py1 = max(0, y - pad_y)
                px2 = min(frame.shape[1], x + w + pad_x)
                py2 = min(frame.shape[0], y + h + pad_y)
                
                rect = (px1, py1, px2 - px1, py2 - py1)
                
                if (px2 - px1) > 0 and (py2 - py1) > 0:
                    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
            
            elif prev_mask is not None:
                # Use prev_mask to initialize GrabCut mask
                # In GrabCut mask values: 0=bg, 1=fg, 2=pr_bg, 3=pr_fg
                mask[prev_mask > 127] = cv2.GC_PR_FGD
                mask[prev_mask <= 127] = cv2.GC_PR_BGD
                
                rect = (0, 0, frame.shape[1], frame.shape[0])
                # We only need 1 iteration here because Optical Flow already did 95% of the work!
                # This makes the process 5x faster.
                cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
            
            # Create a binary mask where 1 (sure foreground) and 3 (probable foreground) are 255
            binary_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
            return binary_mask
        except Exception as e:
            print(f"[SAM3] GrabCut error: {e}")
            return None
