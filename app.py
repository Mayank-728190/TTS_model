import os
import torch
import numpy as np
import io
import soundfile as sf
import logging
from flask import Flask, request, Response, render_template
from torch.serialization import add_safe_globals
from transformers import GenerationConfig  # Import GenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply patching FIRST before any other imports
try:
    import TTS.tts.layers.xtts.gpt as gpt_module
    from transformers import GenerationMixin

    # Save original class reference
    OriginalGPT2InferenceModel = gpt_module.GPT2InferenceModel

    # Create patched class that handles all parameters
    class PatchedGPT2InferenceModel(OriginalGPT2InferenceModel, GenerationMixin):
        def __init__(self, *args, **kwargs):
            # Call original constructor with all parameters
            super().__init__(*args, **kwargs)

            # Required attributes for GenerationMixin
            self.generation_config = GenerationConfig()  # Initialize with default config
            self.main_input_name = "input_ids"

        # Implement required method for GenerationMixin
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            # Create a copy of kwargs to avoid modifying the original
            model_kwargs = kwargs.copy()

            # Remove arguments that shouldn't be passed to the model
            model_kwargs.pop("use_cache", None)
            model_kwargs.pop("past_key_values", None)

            # Call the original method if it exists, otherwise use default behavior
            if hasattr(super(), "prepare_inputs_for_generation"):
                return super().prepare_inputs_for_generation(input_ids, **model_kwargs)

            # Default implementation
            return {"input_ids": input_ids, **model_kwargs}

    # Apply patch to module
    gpt_module.GPT2InferenceModel = PatchedGPT2InferenceModel
    logger.info("✅ Successfully patched GPT2InferenceModel with GenerationMixin")
except Exception as e:
    logger.error(f"❌ Error patching GPT model: {e}")
    raise

# Now import TTS components AFTER patching
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# Torch safety
add_safe_globals([XttsConfig, XttsAudioConfig])
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

app = Flask(__name__)

# === Paths ===
MODEL_DIR = "./model"
FINE_TUNED_PATH = "./fine_tuned_model/xtts_finetuned.pth"
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")

# === Load XTTS model ===
try:
    config = XttsConfig()
    config.load_json(CONFIG_PATH)
    config.model_args.vocab_file = VOCAB_PATH

    tts_model = Xtts.init_from_config(config)
    tts_model.tokenizer = VoiceBpeTokenizer(config.model_args.vocab_file)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    tts_model.load_state_dict(state["model"])
    tts_model.eval()

    # === Load fine-tuned speaker embeddings ===
    latents = torch.load(FINE_TUNED_PATH, map_location="cpu")
    gpt_latent = latents["gpt_latent"]
    speaker_embedding = latents["speaker_embedding"]
    logger.info("✅ XTTS model and fine-tuned speaker loaded.")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    raise

def split_text(text, chunk_size=100):
    """Split text into chunks while preserving sentence boundaries."""
    sentences = []
    current = []
    total_chars = 0

    # Split into sentences while preserving punctuation
    for word in text.split():
        current.append(word)
        total_chars += len(word) + 1  # +1 for space

        # Check if we have a sentence ending
        if word.endswith(('.', '?', '!', ';')):
            if total_chars > chunk_size or len(current) > 15:
                sentences.append(' '.join(current))
                current = []
                total_chars = 0

    # Add any remaining words
    if current:
        sentences.append(' '.join(current))

    # If no natural breaks found, split by chunk size
    if not sentences:
        words = text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    return sentences

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/speak", methods=["POST"])
def speak():
    try:
        data = request.get_json()
        if not data:
            return {"error": "No JSON data received"}, 400

        text = data.get("text", "").strip()
        language = data.get("language", "en")

        if not text:
            return {"error": "Empty input"}, 400

        logger.info(f"Generating speech for: '{text[:50]}...' in {language}")

        audio_chunks = []
        chunks = split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}: '{chunk[:30]}...'")
            with torch.no_grad():
                output = tts_model.inference(
                    text=chunk,
                    language=language,
                    gpt_cond_latent=gpt_latent,
                    speaker_embedding=speaker_embedding,
                    speed=1.0
                )
            audio_chunks.append(output["wav"])

        final_audio = np.concatenate(audio_chunks)
        buffer = io.BytesIO()
        sf.write(buffer, final_audio, 24000, format="WAV")
        buffer.seek(0)

        logger.info("Audio generation complete")
        return Response(buffer.read(), mimetype="audio/wav")

    except Exception as e:
        logger.error(f"Error in speech generation: {str(e)}", exc_info=True)
        return {"error": f"Internal server error: {str(e)}"}, 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    logger.info(f"Starting TTS server on port {port}")
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)
