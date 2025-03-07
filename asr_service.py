import os
import torch
import logging
import time
import requests
import json

from fireredasr.models.fireredasr import FireRedAsr

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FireRedASR model (load only once when module is imported)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models/FireRedASR-AED-L")
ASR_TYPE = "aed"  # Can be "aed" or "llm" based on your preference

# Ollama API URL
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


try:
    model = FireRedAsr.from_pretrained(ASR_TYPE, MODEL_DIR)
    logging.info(f"FireRedASR model loaded: {ASR_TYPE}")
except Exception as e:
    logging.error(f"Failed to load FireRedASR model: {e}")
    raise


def correct_with_ollama(text, audio_info=None):
    """Use Ollama's LLM to correct ASR results"""
    try:
        prompt = f"""你是一个专业的语音识别后处理专家。请校正以下语音识别文本，修正可能的识别错误，添加标点符号，但保持原意。
                    不要删除原文中的任何内容，也不要添加原文中不存在的内容。如果文本完全正确，请直接返回原文。
                    不要添加任何解释或其他格式，只返回校正后的文本。
                    识别文本: {text}
                """

        # if audio_info:
        #     prompt += f"\n音频信息: 时长={audio_info.get('duration', '未知')}秒, RTF={audio_info.get('rtf', '未知')}\n"

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }

        start_time = time.time()
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=300
        )

        response.raise_for_status()
        result = response.json()
        correction_time = time.time() - start_time

        corrected_text = result.get("response", text).strip()
        logging.info(f"Ollama correction completed in {correction_time:.2f}s")

        return corrected_text, correction_time
    except Exception as e:
        logging.error(f"Ollama correction error: {str(e)}")
        return text, 0  # Return original text if correction fails


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

        asr_process_time = time.time() - start_time
        logging.info(f"ASR completed in {asr_process_time:.2f}s: {audio_path}")

        # Correct with Ollama
        audio_info = {
            "duration": float(result.get("rtf", "0").split('/')[-1]) if '/' in result.get("rtf", "0") else 0,
            "rtf": result.get("rtf", "unknown")
        }
        corrected_text, correction_time = correct_with_ollama(result["text"], audio_info)

        total_process_time = asr_process_time + correction_time
        logging.info(f"Total processing in {total_process_time:.2f}s: {audio_path}")

        return {
            "original_transcription": result["text"],
            "corrected_transcription": corrected_text,
            #"transcription": corrected_text,  # For backward compatibility
            "processing_time": total_process_time,
            "asr_time": asr_process_time,
            "correction_time": correction_time,
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
