import cv2
import numpy as np
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, frame_interval=150, target_size=(224, 224), max_frames=50):
        self.frame_interval = frame_interval
        self.target_size = target_size
        self.max_frames = max_frames

    def extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError("Invalid video: No frames detected")

        with tqdm(total=min(total_frames, self.max_frames * self.frame_interval), desc="Extracting frames") as pbar:
            frame_count = 0
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.frame_interval == 0:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, self.target_size)
                        # Normalize to match ResNet-50 expectations
                        frame_normalized = frame_resized / 255.0
                        frame_normalized = (frame_normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                        frames.append((frame_normalized * 255).astype(np.uint8))  # Convert back to uint8
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {str(e)}")
                        continue

                frame_count += 1
                pbar.update(1)

        cap.release()
        print(f"Extracted {len(frames)} frames from {video_path}")
        if not frames:
            raise ValueError("No frames extracted from video")
        return frames
