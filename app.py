from flask import Flask, render_template, request, jsonify
import os
import logging
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor
from content_moderator import ContentModerator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask config
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize processor and moderator
video_processor = VideoProcessor()
content_moderator = ContentModerator(train_mode=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    logger.info("Received video analysis request")
    
    if 'video' not in request.files:
        logger.error("No video file provided")
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        logger.error(f"Invalid file format: {file.filename}")
        return jsonify({'error': 'Invalid file format. Supported formats: mp4, avi, mov, mkv'}), 400

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        logger.error(f"File too large: {file_size} bytes")
        return jsonify({'error': f"File too large. Max size is {app.config['MAX_CONTENT_LENGTH']//1024//1024}MB"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    logger.info(f"File saved to {filepath}")

    try:
        logger.info("Starting frame extraction")
        frames = video_processor.extract_frames(filepath)
        logger.info(f"Extracted {len(frames)} frames")

        logger.info("Starting content moderation")
        results = content_moderator.analyze_frames(frames)
        logger.info("Content moderation completed")

        unsafe_frames = [r for r in results if r['flagged']]
        total_frames = len(results)
        if total_frames > 50:
            results = results[:50]
            unsafe_frames = [r for r in results if r['flagged']]
            total_frames = 50

        unsafe_percentage = (len(unsafe_frames) / total_frames) * 100 if total_frames > 0 else 0

        response = {
            'status': 'UNSAFE' if unsafe_frames else 'SAFE',
            'total_frames': total_frames,
            'unsafe_frames': len(unsafe_frames),
            'unsafe_percentage': unsafe_percentage,
            'confidence': 1.0 if not unsafe_frames else max(r['confidence'] for r in unsafe_frames),
            'details': []
        }

        for frame_idx, result in enumerate(results):
            if result['flagged']:
                response['details'].append({
                    'frame': frame_idx,
                    'reason': result['reason'],
                    'confidence': result['confidence']
                })

        logger.info(f"Analysis result: {response}")
        os.remove(filepath)
        logger.info("Video analysis completed successfully")
        return jsonify(response)

    except ValueError as e:
        logger.error(f"Video processing error: {str(e)}")
        return jsonify({'error': f"Video processing error: {str(e)}"}), 400
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        return jsonify({'error': 'Model file not found. Contact support.'}), 500
    except RuntimeError as e:
        logger.error(f"Model analysis error: {str(e)}")
        return jsonify({'error': f"Model analysis error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False") == "True"
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
