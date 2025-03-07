import os
import tempfile
import requests
from flask import Flask, request, jsonify
from asr_service import process_audio_file

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def recognize_audio():
    # Get audio file URL from request
    request_data = request.get_json()
    if not request_data or 'url' not in request_data:
        return jsonify({'error': 'Missing audio file URL in request'}), 400
    
    audio_url = request_data['url']
    
    # Download the audio file
    try:
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_filename = temp_file.name
        
        with temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
        
        # Process the audio file
        result = process_audio_file(temp_filename)
        
        # Clean up the temporary file
        os.unlink(temp_filename)
        
        # Return the result
        return jsonify(result)
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to download audio file: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing audio file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
