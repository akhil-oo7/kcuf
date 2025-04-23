import cv2
import numpy as np
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, frame_interval=90, target_size=(224, 224), max_frames=150):
        """
        Initialize the VideoProcessor.

        Args:
            frame_interval (int): Number of frames to skip between extractions
            target_size (tuple): Target size for frame resizing (height, width)
            max_frames (int): Maximum number of frames to extract
        """
        self.frame_interval = frame_interval
        self.target_size = target_size
        self.max_frames = max_frames

    def extract_frames(self, video_path):
        """
        Extract frames from a video file.

        Args:
            video_path (str): Path to the video file

        Returns:
            list: List of extracted frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=min(total_frames, self.max_frames * self.frame_interval), desc="Extracting frames") as pbar:
            frame_count = 0
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, self.target_size)
                    frames.append(frame_resized)

                frame_count += 1
                pbar.update(1)

        cap.release()
        print(f"Extracted {len(frames)} frames from {video_path}")
        return frames
