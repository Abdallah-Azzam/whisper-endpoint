"""RunPod Hub requires handler.py at repository root.

Container entrypoint is src/rp_handler.py (see Dockerfile CMD).
This file satisfies Hub static analysis without double-starting the worker.
"""

import runpod


def handler(job):
    from rp_handler import run_whisper_job

    return run_whisper_job(job)


runpod.serverless.start({"handler": handler})
