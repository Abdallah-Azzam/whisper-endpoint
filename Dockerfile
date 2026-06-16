FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

RUN rm -f /etc/apt/sources.list.d/*.list

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV WHISPER_MODEL=large-v3
ENV WHISPER_COMPUTE_TYPE=float16
ENV DEFAULT_ENABLE_VAD=true
ENV DEFAULT_VAD_ONSET=0.5
ENV DEFAULT_VAD_OFFSET=0.363
ENV DEFAULT_SPEECH_PAD_MS=400
ENV DEFAULT_MIN_SILENCE_DURATION_MS=500

WORKDIR /

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends \
        sudo ca-certificates git wget curl bash libgl1 libx11-6 \
        software-properties-common ffmpeg build-essential -y && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && \
    apt-get install python3.10 python3.10-dev python3.10-venv python3-pip -y --no-install-recommends && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install huggingface_hub[hf_xet] && \
    pip install -r /requirements.txt --no-cache-dir

COPY builder/fetch_models.py /fetch_models.py
ARG HF_TOKEN
RUN HF_TOKEN="${HF_TOKEN}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    python /fetch_models.py && rm /fetch_models.py

COPY src .
COPY handler.py .
COPY test_input.json .

CMD python -u /rp_handler.py
