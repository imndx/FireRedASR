import os
import logging
from flask import Flask, request, jsonify
from app import recognize_audio

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/api/v1/speech-to-text', methods=['POST'])
def speech_to_text():
    try:
        return recognize_audio()
    except Exception as e:
        logging.exception("Error in API gateway")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "model": "FireRedASR-AED-L"})

@app.route('/api/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "models": [
            {
                "id": "firered-asr-aed-l",
                "name": "FireRedASR-AED-L",
                "type": "speech-to-text",
                "capabilities": ["mandarin", "chinese-dialects", "english"]
            }
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
