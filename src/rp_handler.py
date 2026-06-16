"""RunPod serverless handler for faster-whisper large-v3."""

import base64
import os
import tempfile
from pathlib import Path

import runpod
from predict import Predictor
from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate

MODEL = Predictor()
MODEL.setup()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _apply_env_defaults(job_input: dict) -> dict:
    resolved = dict(job_input)
    if "model" not in resolved or not resolved.get("model"):
        resolved["model"] = os.environ.get("WHISPER_MODEL", "large-v3")
    if "enable_vad" not in resolved:
        resolved["enable_vad"] = _env_bool("DEFAULT_ENABLE_VAD", True)
    if resolved.get("vad_onset") is None:
        resolved["vad_onset"] = _env_float("DEFAULT_VAD_ONSET", 0.5)
    if resolved.get("vad_offset") is None:
        resolved["vad_offset"] = _env_float("DEFAULT_VAD_OFFSET", 0.363)
    if resolved.get("speech_pad_ms") is None:
        resolved["speech_pad_ms"] = _env_int("DEFAULT_SPEECH_PAD_MS", 400)
    if resolved.get("min_silence_duration_ms") is None:
        resolved["min_silence_duration_ms"] = _env_int(
            "DEFAULT_MIN_SILENCE_DURATION_MS", 500
        )
    return resolved


def base64_to_tempfile(base64_file: str, filename: str | None = None) -> str:
    suffix = Path(filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    job_input = job["input"]

    with rp_debugger.LineTimer("validation_step"):
        input_validation = validate(job_input, INPUT_VALIDATIONS)
        if "errors" in input_validation:
            return {"error": input_validation["errors"]}
        job_input = _apply_env_defaults(input_validation["validated_input"])

    has_audio_url = bool(job_input.get("audio"))
    has_audio_base64 = bool(job_input.get("audio_base64"))
    if not has_audio_url and not has_audio_base64:
        return {"error": "Must provide either audio or audio_base64"}
    if has_audio_url and has_audio_base64:
        return {"error": "Must provide either audio or audio_base64, not both"}

    audio_input = None
    if has_audio_url:
        with rp_debugger.LineTimer("download_step"):
            audio_input = download_files_from_urls(job["id"], [job_input["audio"]])[0]
    else:
        audio_input = base64_to_tempfile(
            job_input["audio_base64"],
            job_input.get("audio_filename"),
        )

    with rp_debugger.LineTimer("prediction_step"):
        whisper_results = MODEL.predict(
            audio=audio_input,
            model_name=job_input["model"],
            transcription=job_input["transcription"],
            translation=job_input["translation"],
            translate=job_input["translate"],
            language=job_input["language"],
            temperature=job_input["temperature"],
            best_of=job_input["best_of"],
            beam_size=job_input["beam_size"],
            patience=job_input["patience"],
            length_penalty=job_input["length_penalty"],
            suppress_tokens=job_input.get("suppress_tokens", "-1"),
            initial_prompt=job_input["initial_prompt"],
            condition_on_previous_text=job_input["condition_on_previous_text"],
            temperature_increment_on_fallback=job_input[
                "temperature_increment_on_fallback"
            ],
            compression_ratio_threshold=job_input["compression_ratio_threshold"],
            logprob_threshold=job_input["logprob_threshold"],
            no_speech_threshold=job_input["no_speech_threshold"],
            enable_vad=job_input["enable_vad"],
            vad_onset=job_input["vad_onset"],
            vad_offset=job_input["vad_offset"],
            speech_pad_ms=job_input["speech_pad_ms"],
            min_silence_duration_ms=job_input["min_silence_duration_ms"],
            word_timestamps=job_input["word_timestamps"],
        )

    with rp_debugger.LineTimer("cleanup_step"):
        rp_cleanup.clean(["input_objects"])

    return whisper_results
