from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoFrameDataset(Dataset):
    def __init__(self, frames, labels, feature_extractor):
        self.frames = frames
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        image = Image.fromarray(frame)
        inputs = self.feature_extractor(image, return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ContentModerator:
    def __init__(self, model_name="microsoft/resnet-50", train_mode=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model_name = model_name
        self.threshold = float(os.environ.get("VIOLENCE_THRESHOLD", 0.6))

        logger.info(f"Loading feature extractor for {model_name}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        if train_mode:
            logger.info("Initializing model for training")
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=2,
                ignore_mismatched_sizes=True
            ).to(self.device)
        else:
            model_path = os.path.join("models", "best_model")
            logger.info(f"Loading trained model from {model_path}")
            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}")
                raise FileNotFoundError(f"Model not found at {model_path}. Please ensure models/best_model exists.")
            self.model = AutoModelForImageClassification.from_pretrained(
                model_path,
                num_labels=2
            ).to(self.device)
            self.model.eval()

    def analyze_frames(self, frames):
        logger.info(f"Analyzing {len(frames)} frames")
        results = []
        try:
            dataset = VideoFrameDataset(frames, [0] * len(frames), self.feature_extractor)
            dataloader = DataLoader(dataset, batch_size=8)
            with torch.no_grad():
                for batch in dataloader:
                    pixel_values = batch['pixel_values'].to(self.device)
                    outputs = self.model(pixel_values)
                    predictions = torch.softmax(outputs.logits, dim=1)

                    for pred in predictions:
                        violence_prob = pred[1].item()
                        flagged = violence_prob > self.threshold
                        results.append({
                            'flagged': flagged,
                            'reason': "Detected violence" if flagged else "No inappropriate content detected",
                            'confidence': violence_prob if flagged else 1 - violence_prob
                        })
                        logger.debug(f"Frame prob: {violence_prob}, Flagged: {flagged}")
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            raise

        logger.info("Frame analysis completed")
        return results
