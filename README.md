# Whisper Endpoint (RunPod Serverless)

Custom RunPod Serverless worker for **faster-whisper `large-v3`** with Silero VAD enabled by default. Based on the [official RunPod faster-whisper worker](https://github.com/runpod-workers/worker-faster_whisper), optimized for MedScribe-AI transcription.

## Features

- Pre-baked `large-v3` model (smaller image, faster cold start)
- VAD enabled by default with tunable Silero `onset` / `offset`
- Official RunPod input contract plus VAD extension fields
- Supports `audio` URL or `audio_base64` (WebM/WAV via ffmpeg)

## API

**Endpoint:** `POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync`

**Request:**

```json
{
  "input": {
    "audio_base64": "<base64-encoded audio>",
    "model": "large-v3",
    "enable_vad": true,
    "vad_onset": 0.5,
    "vad_offset": 0.363,
    "speech_pad_ms": 400,
    "min_silence_duration_ms": 500,
    "temperature": 0,
    "language": "en"
  }
}
```

Alternatively, pass `"audio": "https://..."` instead of `audio_base64`.

**Response:**

```json
{
  "status": "COMPLETED",
  "output": {
    "segments": [{"text": "...", "start": 0.0, "end": 1.2}],
    "transcription": "full plain text",
    "detected_language": "en",
    "model": "large-v3"
  }
}
```

### VAD parameters

| Field | Default | Description |
|-------|---------|-------------|
| `enable_vad` | `true` | Use Silero VAD to skip non-speech |
| `vad_onset` | `0.5` | Speech probability threshold to start a segment |
| `vad_offset` | `0.363` | Silence threshold to end a segment |
| `speech_pad_ms` | `400` | Padding around detected speech |
| `min_silence_duration_ms` | `500` | Minimum silence before splitting segments |

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `WHISPER_MODEL` | `large-v3` | Model baked and loaded at startup |
| `WHISPER_COMPUTE_TYPE` | `float16` | GPU compute type |
| `DEFAULT_ENABLE_VAD` | `true` | Default VAD when input omits `enable_vad` |
| `DEFAULT_VAD_ONSET` | `0.5` | Default VAD onset |
| `DEFAULT_VAD_OFFSET` | `0.363` | Default VAD offset |
| `DEFAULT_SPEECH_PAD_MS` | `400` | Default speech padding |
| `DEFAULT_MIN_SILENCE_DURATION_MS` | `500` | Default silence split threshold |

## Build and deploy

```bash
docker build -t your-registry/whisper-endpoint:latest .
docker push your-registry/whisper-endpoint:latest
```

In RunPod Serverless:

1. Create endpoint from custom Docker image
2. GPU: RTX A4000 / A5000 / L4 (16 GB VRAM)
3. Container disk: 15 GB+
4. Set env vars above (optional; defaults are sensible)

**Smoke test:**

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"audio":"https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav","model":"large-v3","enable_vad":true,"vad_onset":0.5,"vad_offset":0.363}}'
```

## Local test

Requires CUDA GPU and downloaded model weights (inside the built container):

```bash
python /test_local.py
```

## MedScribe integration

Set in MedScribe backend / ARQ worker:

```
STT_PROVIDER=runpod
RUNPOD_API_KEY=rpa_...
RUNPOD_STT_ENDPOINT_ID=<your-endpoint-id>
RUNPOD_STT_VAD_ONSET=0.500
RUNPOD_STT_VAD_OFFSET=0.363
```
