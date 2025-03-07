import os
import tempfile
import requests
import subprocess
import logging
from flask import Flask, request, jsonify
from asr_service import process_audio_file
from urllib.parse import urlparse

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def is_wav_file(filename):
    """Check if the file is already in WAV format based on extension."""
    return filename.lower().endswith('.wav')

def convert_to_wav(input_file, output_file):
    """Convert audio file to 16kHz 16-bit mono WAV using ffmpeg."""
    try:
        cmd = [
            'ffmpeg',
            '-i', input_file,  # Input file
            '-ar', '16000',    # Sample rate: 16kHz
            '-ac', '1',        # Audio channels: mono
            '-acodec', 'pcm_s16le',  # Encoding: 16-bit PCM
            '-y',              # Overwrite output file if exists
            output_file        # Output WAV file
        ]
        
        process = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode != 0:
            logging.error(f"FFmpeg conversion error: {process.stderr}")
            return False
        
        return True
    except Exception as e:
        logging.error(f"Error converting audio: {str(e)}")
        return False

@app.route('/recognize', methods=['POST'])
def recognize_audio():
    # Get audio file URL from request
    request_data = request.get_json()
    if not request_data or 'url' not in request_data:
        return jsonify({'error': 'Missing audio file URL in request'}), 400
    
    audio_url = request_data['url']
    
    # Download the audio file
    try:
        # Get the original filename from URL for extension detection
        parsed_url = urlparse(audio_url)
        original_filename = os.path.basename(parsed_url.path)
        
        # Download file to a temporary location
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()
        
        # Save the downloaded file (in its original format)
        temp_original = tempfile.NamedTemporaryFile(delete=False)
        temp_original_path = temp_original.name
        
        with temp_original:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_original.write(chunk)
        
        # Prepare the WAV file path
        temp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        
        # Check if conversion is needed
        if is_wav_file(original_filename):
            logging.info(f"File is already in WAV format: {original_filename}")
            temp_wav_path = temp_original_path
        else:
            logging.info(f"Converting {original_filename} to WAV format")
            if not convert_to_wav(temp_original_path, temp_wav_path):
                os.unlink(temp_original_path)
                if temp_original_path != temp_wav_path:
                    try:
                        os.unlink(temp_wav_path)
                    except:
                        pass
                return jsonify({'error': 'Failed to convert audio to WAV format'}), 500
        
        # Process the audio file
        result = process_audio_file(temp_wav_path)
        
        # Clean up temporary files
        if temp_original_path != temp_wav_path:
            os.unlink(temp_original_path)
        os.unlink(temp_wav_path)
        
        # Return the result
        return jsonify(result)
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to download audio file: {str(e)}'}), 400
    except Exception as e:
        logging.exception("Error processing request")
        return jsonify({'error': f'Error processing audio file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
