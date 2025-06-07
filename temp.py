import os
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.models.xtts import XttsAudioConfig  # ✅ Import this
from torch.serialization import add_safe_globals

# Fix PyTorch pickle security
add_safe_globals([XttsConfig, XttsAudioConfig])  # ✅ Added missing safe global
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

# === Paths ===
MODEL_DIR = "./model"
REFERENCE_PATH = "./speaker.wav"
OUTPUT_PATH = "./fine_tuned_model/xtts_finetuned.pth"

CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
SPEAKER_PATH = os.path.join(MODEL_DIR, "speakers_xtts.pth")

# Load config and model
config = XttsConfig()
config.load_json(CONFIG_PATH)
config.model_args.vocab_file = VOCAB_PATH

tts_model = Xtts.init_from_config(config)
tts_model.tokenizer = VoiceBpeTokenizer(config.model_args.vocab_file)

# ✅ Use weights_only=False
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
tts_model.load_state_dict(state["model"])
tts_model.eval()

# Generate speaker embedding and latent
with torch.no_grad():
    gpt_latent, speaker_embedding = tts_model.get_conditioning_latents(
        audio_path=REFERENCE_PATH,
        gpt_cond_len=3
    )

# Save them
os.makedirs("fine_tuned_model", exist_ok=True)
torch.save({
    "config": config.to_dict(),
    "gpt_latent": gpt_latent,
    "speaker_embedding": speaker_embedding
}, OUTPUT_PATH)

print(f"✅ Saved fine-tuned conditioning latents to {OUTPUT_PATH}")
