from flask import Flask, render_template, request, jsonify
import os
import logging
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor
from content_moderator import ContentModerator
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

video_processor = VideoProcessor()
content_moderator = ContentModerator(train_mode=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    logger.info("Video analysis request received.")

    if 'video' not in request.files or request.files['video'].filename == '':
        logger.error("No video file provided.")
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file format. Supported: mp4, avi, mov, mkv'}), 400

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': f"File too large. Max size: {app.config['MAX_CONTENT_LENGTH']//1024//1024}MB"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    logger.info(f"File saved at {filepath}")

    try:
        # Extract frames
        frames = video_processor.extract_frames(filepath)
        results = content_moderator.analyze_frames(frames)

        # Analyze and prepare the response
        unsafe_frames = [r for r in results if r['flagged']]
        total_frames = len(results)
        results = results[:50]  # truncate for UI
        unsafe_percentage = (len(unsafe_frames) / total_frames) * 100 if total_frames > 0 else 0

        response = {
            'status': 'UNSAFE' if unsafe_frames else 'SAFE',
            'total_frames': total_frames,
            'unsafe_frames': len(unsafe_frames),
            'unsafe_percentage': unsafe_percentage,
            'confidence': 1.0 if not unsafe_frames else max(r['confidence'] for r in unsafe_frames),
            'details': [
                {
                    'frame': idx,
                    'reason': r['reason'],
                    'confidence': r['confidence']
                } for idx, r in enumerate(results) if r['flagged']
            ]
        }

        logger.info("Analysis complete")
        os.remove(filepath)
        return jsonify(response)

    except ValueError as e:
        logger.error(str(e))
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError as e:
        logger.error(str(e))
        return jsonify({'error': 'Model file not found. Contact support.'}), 500
    except RuntimeError as e:
        logger.error(str(e))
        return jsonify({'error': f"Model analysis error: {e}"}), 500
    except Exception as e:
        logger.error(str(e))
        return jsonify({'error': f"Unexpected error: {e}"}), 500

if __name__ == '__main__':
    # Force CPU usage to avoid GPU/segmentation faults.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Configure port and debug mode
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False") == "True"
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
