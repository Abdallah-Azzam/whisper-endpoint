"""Faster-whisper predictor with Silero VAD support."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.utils import format_timestamp
from runpod.serverless.utils import rp_cuda

DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DEFAULT_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
DEFAULT_ENABLE_VAD = os.environ.get("DEFAULT_ENABLE_VAD", "true").lower() in (
    "1",
    "true",
    "yes",
)
DEFAULT_VAD_ONSET = float(os.environ.get("DEFAULT_VAD_ONSET", "0.5"))
DEFAULT_VAD_OFFSET = float(os.environ.get("DEFAULT_VAD_OFFSET", "0.363"))
DEFAULT_SPEECH_PAD_MS = int(os.environ.get("DEFAULT_SPEECH_PAD_MS", "400"))
DEFAULT_MIN_SILENCE_DURATION_MS = int(
    os.environ.get("DEFAULT_MIN_SILENCE_DURATION_MS", "500")
)


class Predictor:
    """Single-model faster-whisper predictor."""

    def __init__(self) -> None:
        self.model: Optional[WhisperModel] = None
        self.model_name: Optional[str] = None

    def setup(self) -> None:
        self._load_model(DEFAULT_MODEL)

    def _load_model(self, model_name: str) -> WhisperModel:
        if self.model is not None and self.model_name == model_name:
            return self.model

        device = "cuda" if rp_cuda.is_available() else "cpu"
        compute_type = (
            DEFAULT_COMPUTE_TYPE
            if device == "cuda"
            else "int8"
        )
        print(f"Loading model {model_name} on {device} ({compute_type})...")
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        self.model_name = model_name
        print(f"Model {model_name} loaded.")
        return self.model

    def predict(
        self,
        audio: str,
        model_name: str = DEFAULT_MODEL,
        transcription: str = "plain_text",
        translate: bool = False,
        translation: str = "plain_text",
        language: Optional[str] = None,
        temperature: float = 0,
        best_of: int = 5,
        beam_size: int = 5,
        patience: float = 1.0,
        length_penalty: Optional[float] = None,
        suppress_tokens: str = "-1",
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
        temperature_increment_on_fallback: float = 0.2,
        compression_ratio_threshold: float = 2.4,
        logprob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        enable_vad: Optional[bool] = None,
        vad_onset: Optional[float] = None,
        vad_offset: Optional[float] = None,
        speech_pad_ms: Optional[int] = None,
        min_silence_duration_ms: Optional[int] = None,
        word_timestamps: bool = False,
    ) -> Dict[str, Any]:
        model = self._load_model(model_name)

        if enable_vad is None:
            enable_vad = DEFAULT_ENABLE_VAD

        if temperature_increment_on_fallback is not None:
            temperature_values = tuple(
                np.arange(
                    temperature,
                    1.0 + 1e-6,
                    temperature_increment_on_fallback,
                )
            )
        else:
            temperature_values = [temperature]

        vad_parameters = None
        if enable_vad:
            vad_parameters = {
                "onset": vad_onset if vad_onset is not None else DEFAULT_VAD_ONSET,
                "offset": vad_offset if vad_offset is not None else DEFAULT_VAD_OFFSET,
                "speech_pad_ms": (
                    speech_pad_ms
                    if speech_pad_ms is not None
                    else DEFAULT_SPEECH_PAD_MS
                ),
                "min_silence_duration_ms": (
                    min_silence_duration_ms
                    if min_silence_duration_ms is not None
                    else DEFAULT_MIN_SILENCE_DURATION_MS
                ),
            }

        segments_iter, info = model.transcribe(
            str(audio),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            temperature=temperature_values,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            suppress_blank=True,
            suppress_tokens=[-1],
            without_timestamps=False,
            max_initial_timestamp=1.0,
            word_timestamps=word_timestamps,
            vad_filter=enable_vad,
            vad_parameters=vad_parameters,
        )
        segments = list(segments_iter)

        transcription_output = format_segments(transcription, segments)

        translation_output = None
        if translate:
            translation_segments, _ = model.transcribe(
                str(audio),
                task="translate",
                temperature=temperature_values,
                vad_filter=enable_vad,
                vad_parameters=vad_parameters,
            )
            translation_output = format_segments(
                translation, list(translation_segments)
            )

        results: Dict[str, Any] = {
            "segments": serialize_segments(segments),
            "detected_language": info.language,
            "transcription": transcription_output,
            "translation": translation_output,
            "device": "cuda" if rp_cuda.is_available() else "cpu",
            "model": model_name,
        }

        if word_timestamps:
            word_timestamps_list: List[Dict[str, Any]] = []
            for segment in segments:
                if segment.words is None:
                    continue
                for word in segment.words:
                    word_timestamps_list.append(
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                        }
                    )
            results["word_timestamps"] = word_timestamps_list

        return results


def serialize_segments(transcript: List[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "id": segment.id,
            "seek": segment.seek,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "tokens": segment.tokens,
            "temperature": segment.temperature,
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob,
        }
        for segment in transcript
    ]


def format_segments(format_type: str, segments: List[Any]) -> str:
    if format_type == "plain_text":
        return " ".join(segment.text.lstrip() for segment in segments)
    if format_type == "formatted_text":
        return "\n".join(segment.text.lstrip() for segment in segments)
    if format_type == "srt":
        return write_srt(segments)
    if format_type == "vtt":
        return write_vtt(segments)
    return " ".join(segment.text.lstrip() for segment in segments)


def write_vtt(transcript: List[Any]) -> str:
    result = ""
    for segment in transcript:
        result += (
            f"{format_timestamp(segment.start, always_include_hours=True)} --> "
            f"{format_timestamp(segment.end, always_include_hours=True)}\n"
        )
        result += f"{segment.text.strip().replace('-->', '->')}\n\n"
    return result


def write_srt(transcript: List[Any]) -> str:
    result = ""
    for index, segment in enumerate(transcript, start=1):
        result += f"{index}\n"
        result += (
            f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        )
        result += f"{segment.text.strip().replace('-->', '->')}\n\n"
    return result
