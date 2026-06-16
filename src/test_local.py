"""Run the handler locally without RunPod (requires GPU + model weights)."""
import json
from pathlib import Path

from rp_handler import run_whisper_job


def main() -> None:
    payload = json.loads(Path("/test_input.json").read_text(encoding="utf-8"))
    result = run_whisper_job({"id": "local-test", "input": payload["input"]})
    print(json.dumps(result, indent=2)[:2000])


if __name__ == "__main__":
    main()
