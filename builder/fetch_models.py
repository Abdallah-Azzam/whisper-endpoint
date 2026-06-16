"""Pre-download Whisper model weights at image build time."""
import os

from faster_whisper.utils import download_model

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")

print(f"Downloading {MODEL_NAME}...")
download_model(MODEL_NAME, cache_dir=None)
print(f"Finished downloading {MODEL_NAME}.")
