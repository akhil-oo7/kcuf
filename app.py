from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor
from content_moderator import ContentModerator
from dotenv import load_dotenv

load_dotenv()

port = int(os.environ.get("PORT", 5000))
debug_mode = os.environ.get("DEBUG", "False") == "True"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

video_processor = VideoProcessor()  # Optimized processor
content_moderator = ContentModerator(train_mode=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file format. Supported formats: mp4, avi, mov, mkv'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            print(f"Uploaded file saved to {filepath}")
            frames = video_processor.extract_frames(filepath)
            results = content_moderator.analyze_frames(frames)

            unsafe_frames = [r for r in results if r['flagged']]
            total_frames = len(results)
            unsafe_percentage = (len(unsafe_frames) / total_frames) * 100

            response = {
                'status': 'UNSAFE' if unsafe_frames else 'SAFE',
                'total_frames': total_frames,
                'unsafe_frames': len(unsafe_frames),
                'unsafe_percentage': unsafe_percentage,
                'confidence': 1.0 if not unsafe_frames else max(r['confidence'] for r in unsafe_frames),
                'details': []
            }

            if unsafe_frames:
                for frame_idx, result in enumerate(results):
                    if result['flagged']:
                        response['details'].append({
                            'frame': frame_idx,
                            'reason': result['reason'],
                            'confidence': result['confidence']
                        })

            os.remove(filepath)
            return jsonify(response)

        except ValueError as e:
            return jsonify({'error': f"Video processing error: {str(e)}"}), 400
        except RuntimeError as e:
            return jsonify({'error': f"Model analysis error: {str(e)}"}), 500
        except Exception as e:
            return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
