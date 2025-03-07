import os
import torch
import logging
import time

from fireredasr.models.fireredasr import FireRedAsr

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FireRedASR model (load only once when module is imported)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models/FireRedASR-AED-L")
ASR_TYPE = "aed"  # Can be "aed" or "llm" based on your preference

try:
    model = FireRedAsr.from_pretrained(ASR_TYPE, MODEL_DIR)
    logging.info(f"FireRedASR model loaded: {ASR_TYPE}")
except Exception as e:
    logging.error(f"Failed to load FireRedASR model: {e}")
    raise

def process_audio_file(audio_path):
    """Process a single audio file and return recognition results."""
    try:
        start_time = time.time()
        logging.info(f"Processing file: {audio_path}")
        
        # Process using FireRedASR
        uttid = os.path.basename(audio_path).rsplit('.', 1)[0]  # Use filename as uttid
        batch_uttid = [uttid]
        batch_wav_path = [audio_path]
        
        # Use GPU if available
        args = {
            "use_gpu": torch.cuda.is_available(),
            "beam_size": 3,
            "nbest": 1,
            "decode_max_len": 0,
            "softmax_smoothing": 1.25,
            "aed_length_penalty": 0.6,
            "eos_penalty": 1.0
        }
        
        results = model.transcribe(batch_uttid, batch_wav_path, args)
        
        # Extract result
        result = results[0]  # First item since we only sent one audio file
        
        process_time = time.time() - start_time
        logging.info(f"Processed in {process_time:.2f}s: {audio_path}")
        
        return {
            "transcription": result["text"],
            "processing_time": process_time,
            "uttid": result["uttid"],
            "rtf": result["rtf"],
            "status": "success"
        }
        
    except Exception as e:
        logging.error(f"Error processing {audio_path}: {str(e)}")
        return {
            "error": str(e),
            "status": "error"
        }
