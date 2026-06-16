# Whisper Endpoint

[![Runpod](https://api.runpod.io/badge/Abdallah-Azzam/whisper-endpoint)](https://console.runpod.io/hub/Abdallah-Azzam/whisper-endpoint)

RunPod Serverless worker for **faster-whisper `large-v3`** with Silero VAD enabled by default.

## Deploy

Click **Deploy on RunPod** above, then choose a GPU (16 GB VRAM recommended, e.g. RTX A4000 / L4) and deploy. VAD defaults can be adjusted in the endpoint environment settings.

## Features

- Pre-baked `large-v3` model
- Silero VAD on by default (`vad_onset` / `vad_offset` tunable per request)
- Accepts audio via URL or base64 (WebM, WAV, and other ffmpeg-supported formats)

## API

`POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync`

**Request:**

```json
{
  "input": {
    "audio": "https://example.com/audio.wav",
    "model": "large-v3",
    "enable_vad": true,
    "vad_onset": 0.5,
    "vad_offset": 0.363,
    "temperature": 0
  }
}
```

Use `audio_base64` instead of `audio` to send encoded file bytes. Optional `language` (e.g. `"en"`) forces a language; omit for auto-detect.

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

### Input parameters

| Field | Default | Description |
|-------|---------|-------------|
| `enable_vad` | `true` | Skip non-speech with Silero VAD |
| `vad_onset` | `0.5` | Speech onset threshold |
| `vad_offset` | `0.363` | Speech offset threshold |
| `speech_pad_ms` | `400` | Padding around speech segments |
| `min_silence_duration_ms` | `500` | Silence duration before splitting |
| `temperature` | `0` | Whisper decoding temperature |
| `language` | auto | ISO language code (optional) |

### Endpoint environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `WHISPER_MODEL` | `large-v3` | Loaded model |
| `WHISPER_COMPUTE_TYPE` | `float16` | GPU compute type |
| `DEFAULT_ENABLE_VAD` | `true` | Default when request omits `enable_vad` |
| `DEFAULT_VAD_ONSET` | `0.5` | Default VAD onset |
| `DEFAULT_VAD_OFFSET` | `0.363` | Default VAD offset |

## Example

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"audio":"https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav","model":"large-v3","enable_vad":true}}'
```
