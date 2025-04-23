import os
from video_processor import VideoProcessor
import numpy as np
from tqdm import tqdm

class VideoDataset:
    def __init__(self, violence_dir, nonviolence_dir, frame_interval=30):
        """
        Initialize the video dataset.
        
        Args:
            violence_dir (str): Directory containing violent videos
            nonviolence_dir (str): Directory containing non-violent videos
            frame_interval (int): Number of frames to skip between extractions
        """
        self.violence_dir = violence_dir
        self.nonviolence_dir = nonviolence_dir
        self.frame_interval = frame_interval
        self.video_processor = VideoProcessor(frame_interval)
        
        self.violence_videos = [os.path.join(violence_dir, f) for f in os.listdir(violence_dir) 
                              if f.endswith(('.mp4', '.avi', '.mov'))]
        self.nonviolence_videos = [os.path.join(nonviolence_dir, f) for f in os.listdir(nonviolence_dir) 
                                 if f.endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"Found {len(self.violence_videos)} violent videos")
        print(f"Found {len(self.nonviolence_videos)} non-violent videos")
    
    def process_videos(self, max_videos_per_class=None):
        """
        Process videos and extract frames.
        
        Args:
            max_videos_per_class (int): Maximum number of videos to process per class
            
        Returns:
            tuple: (frames, labels) where frames is a list of numpy arrays and labels is a list of 0/1
        """
        frames = []
        labels = []
        
        for video_path in tqdm(self.violence_videos[:max_videos_per_class], desc="Processing violent videos"):
            try:
                video_frames = self.video_processor.extract_frames(video_path)
                frames.extend(video_frames)
                labels.extend([1] * len(video_frames))
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
        
        for video_path in tqdm(self.nonviolence_videos[:max_videos_per_class], desc="Processing non-violent videos"):
            try:
                video_frames = self.video_processor.extract_frames(video_path)
                frames.extend(video_frames)
                labels.extend([0] * len(video_frames))
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
        
        return np.array(frames), np.array(labels)