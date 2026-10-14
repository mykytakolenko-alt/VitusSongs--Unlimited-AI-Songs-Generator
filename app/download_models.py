#!/usr/bin/env python3
"""Download YuE2 model weights from Hugging Face into models/ (resumable).

Prints machine-readable progress lines for the UI:
  STEP <repo>            current repository
  PROGRESS <done> <total> <bytes_per_second>
  ERROR <message>        fatal error
"""
import os
import shutil
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPOS = [("m-a-p/YuE2-3B", "YuE2-3B"), ("m-a-p/YuE2-Vae", "YuE2-Vae")]


def dir_bytes(path: Path) -> int:
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def main() -> int:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("ERROR huggingface_hub is not installed. Run install.bat.", flush=True)
        return 1

    print("STEP Contacting Hugging Face…", flush=True)
    total = 0
    try:
        api = HfApi()
        for repo, _ in REPOS:
            info = api.model_info(repo, files_metadata=True)
            total += sum((s.size or 0) for s in info.siblings)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR Cannot reach Hugging Face ({exc}). Check your internet connection.", flush=True)
        return 1

    MODELS.mkdir(exist_ok=True)
    need = max(0, total - dir_bytes(MODELS)) + 2**30
    free = shutil.disk_usage(MODELS).free
    if free < need:
        print(f"ERROR Not enough free disk space: about {need / 2**30:.1f} GB needed, "
              f"{free / 2**30:.1f} GB free. Free some space and press Resume.", flush=True)
        return 1
    failure = []
    finished = threading.Event()

    def work():
        try:
            for repo, name in REPOS:
                print(f"STEP Downloading {repo}", flush=True)
                snapshot_download(repo_id=repo, local_dir=str(MODELS / name), max_workers=4)
        except BaseException as exc:  # noqa: BLE001
            failure.append(exc)
        finally:
            finished.set()

    threading.Thread(target=work, daemon=True).start()

    last_bytes, last_time, speed = dir_bytes(MODELS), time.time(), 0.0
    while not finished.wait(1.5):
        now, cur = time.time(), dir_bytes(MODELS)
        dt = now - last_time
        if dt > 0:
            speed = 0.6 * speed + 0.4 * max(0.0, (cur - last_bytes) / dt)
        last_bytes, last_time = cur, now
        print(f"PROGRESS {min(cur, total)} {total} {speed:.0f}", flush=True)

    if failure:
        print(f"ERROR {failure[0]}", flush=True)
        return 1
    print(f"PROGRESS {total} {total} 0", flush=True)
    print("STEP All models downloaded.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
