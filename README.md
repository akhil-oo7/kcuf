# Video Content Moderation System

This project implements a video content moderation system using Large Language Models (LLMs) and computer vision techniques to detect inappropriate content in videos.

## Features

- Video frame extraction
- Content analysis using pre-trained models
- Detection of inappropriate content categories:
  - Violence
  - Nudity
  - Drugs
  - Hate speech
- Confidence scoring for detected content

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python main.py
```

2. When prompted, enter the path to the video file you want to analyze.

3. The system will process the video and display results for any potentially inappropriate content found.

## How it Works

1. The system extracts frames from the video at regular intervals
2. Each frame is analyzed using a pre-trained image classification model
3. The model's predictions are checked against predefined categories of inappropriate content
4. Results are displayed showing any flagged content with confidence scores

## Customization

You can customize the content moderation by:

1. Modifying the `inappropriate_categories` dictionary in `content_moderator.py`
2. Adjusting the frame extraction interval in `video_processor.py`
3. Using a different pre-trained model by changing the `model_name` parameter in `ContentModerator`

## License

MIT License 