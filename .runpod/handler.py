"""RunPod Hub handler (see .runpod/hub.json)."""

import sys
from pathlib import Path

# Repo layout: implementation lives in src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import runpod
from rp_handler import run_whisper_job

runpod.serverless.start({"handler": run_whisper_job})
