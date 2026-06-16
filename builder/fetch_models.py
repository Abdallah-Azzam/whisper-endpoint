"""Pre-download Whisper model weights at image build time."""
import os

from faster_whisper.utils import download_model

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

if HF_TOKEN:
    from huggingface_hub import login

    login(token=HF_TOKEN, add_to_git_credential=False)
    print("Authenticated with Hugging Face Hub.")
else:
    print("No HF_TOKEN set; downloading as anonymous (may be slower).")

print(f"Downloading {MODEL_NAME}...")
download_model(MODEL_NAME, cache_dir=None)
print(f"Finished downloading {MODEL_NAME}.")
