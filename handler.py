"""RunPod Serverless entrypoint (required by RunPod Hub)."""

import runpod

from rp_handler import run_whisper_job

runpod.serverless.start({"handler": run_whisper_job})
