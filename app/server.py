#!/usr/bin/env python3
"""VitusSongs — local web UI for AI song generation with YuE2.

Standard library only (audio editing lazily imports numpy + soundfile). Each generation
runs as a separate process (`python -m yue2.cli generate`), so GPU memory is fully released
after every song and cancelling is instant.
"""
from __future__ import annotations

import ctypes
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "app" / "static"
OUTPUTS = Path(os.environ.get("YUE_STUDIO_OUTPUTS", ROOT / "outputs"))
MODELS = ROOT / "models"
JOBS = ROOT / "app" / "_jobs"
UPLOADS = OUTPUTS / "_uploads"
PROBE_CACHE = ROOT / "app" / "_probe_cache.json"
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".opus"}
MAX_UPLOAD = 600 * 2**20
HOST, PORT = "127.0.0.1", int(os.environ.get("YUE_STUDIO_PORT", "7860"))
CLI_MODULE = os.environ.get("YUE_STUDIO_CLI", "yue2.cli")
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}

for _d in (OUTPUTS, MODELS, JOBS, UPLOADS):
    _d.mkdir(parents=True, exist_ok=True)

mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("audio/mpeg", ".mp3")
mimetypes.add_type("image/svg+xml", ".svg")

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
BELOW_NORMAL = 0x00004000 if os.name == "nt" else 0  # keep Windows responsive during long jobs

EDIT_LOCK = threading.Lock()  # one heavy audio operation at a time
LAST = {"ping": time.time() + 90, "bye": 0.0}  # window heartbeat (90 s grace at startup)


class State:
    """Single queue: one generation at a time (YuE2 requirement)."""

    def __init__(self):
        self.lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.job: dict | None = None
        self.log: list[str] = []
        self.dl_proc: subprocess.Popen | None = None
        self.dl_log: list[str] = []
        self.dl = {"bytes": 0, "total": 0, "speed": 0.0, "step": "", "error": None}

    def add_log(self, line: str):
        with self.lock:
            self.log.append(line)
            if len(self.log) > 3000:
                del self.log[:1000]
                if self.job is not None:
                    self.job["log_offset"] = self.job.get("log_offset", 0) + 1000

    def generating(self) -> bool:
        with self.lock:
            return bool(self.job and self.job.get("status") == "running")

    def downloading(self) -> bool:
        with self.lock:
            return bool(self.dl_proc and self.dl_proc.poll() is None)


S = State()


# ---------------------------------------------------------------- environment / helpers
def python_exe() -> str:
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        if (ROOT / rel).exists():
            return str(ROOT / rel)
    return sys.executable


def child_env() -> dict:
    env = dict(os.environ)
    env["YUE2_KIT"] = str(ROOT)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
               HF_HUB_DISABLE_SYMLINKS_WARNING="1", HF_HUB_DISABLE_TELEMETRY="1",
               HF_HUB_DISABLE_PROGRESS_BARS="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    # Physical cores only (SMT siblings slow down matrix math) and less allocator fragmentation.
    threads = os.cpu_count() or 2
    cores = max(1, threads // 2) if threads >= 8 else threads
    env.setdefault("OMP_NUM_THREADS", str(cores))
    env.setdefault("MKL_NUM_THREADS", str(cores))
    alloc = "garbage_collection_threshold:0.8,max_split_size_mb:128"
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        env.setdefault(key, alloc)
    return env


def slug(text: str) -> str:
    text = re.sub(r"[^\w\- ]+", "", text, flags=re.UNICODE).strip().replace(" ", "_")
    return text[:40] or "song"


def kill_tree(proc: subprocess.Popen | None):
    if proc is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=CREATE_NO_WINDOW, timeout=15)
        else:
            proc.terminate()
    except Exception:
        pass


def rmtree_retry(path: Path, tries: int = 6):
    """Windows keeps files locked for a moment after a process dies; retry a few times."""
    for _ in range(tries):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return
        time.sleep(0.5)


def free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 1 << 62


def keep_awake(on: bool):
    if os.name != "nt":
        return
    try:  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED (applies to the calling thread)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if on else 0x80000000)  # type: ignore[attr-defined]
    except Exception:
        pass


_MODELS_CACHE = {"t": 0.0, "v": {"model": False, "vae": False}}


def models_ready(force: bool = False) -> dict:
    """Cached for a few seconds: the UI polls this constantly."""
    now = time.time()
    if not force and now - _MODELS_CACHE["t"] < 4:
        return _MODELS_CACHE["v"]

    def ok(p: Path) -> bool:
        if not p.is_dir():
            return False
        cache = p / ".cache"
        if cache.exists() and next(cache.rglob("*.incomplete"), None):
            return False
        return any(next(p.glob(ext), None) for ext in ("*.safetensors", "*.bin", "*.pt", "*.pth"))

    _MODELS_CACHE.update(t=now, v={"model": ok(MODELS / "YuE2-3B"), "vae": ok(MODELS / "YuE2-Vae")})
    return _MODELS_CACHE["v"]


# ---------------------------------------------------------------- accelerator probe (cached)
_PROBE: dict = {"pending": True, "name": None}


def _torch_stamp() -> str | None:
    for pattern in (".venv/Lib/site-packages/torch/version.py", ".venv/lib/python*/site-packages/torch/version.py"):
        for f in ROOT.glob(pattern):
            return str(int(f.stat().st_mtime))
    return None


def run_probe():
    """Ask PyTorch (in a separate process) which accelerator exists. Positive results are cached."""
    global _PROBE
    stamp = _torch_stamp()
    try:
        cached = json.loads(PROBE_CACHE.read_text(encoding="utf-8"))
        if (stamp and cached.get("stamp") == stamp and time.time() - cached.get("t", 0) < 6 * 3600
                and cached["data"].get("backend") in ("cuda", "rocm")):
            _PROBE = cached["data"]
            return
    except Exception:
        pass
    try:
        out = subprocess.run([python_exe(), str(ROOT / "app" / "probe.py")], capture_output=True, text=True,
                             timeout=240, env=child_env(), creationflags=CREATE_NO_WINDOW)
        data = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        data = {"pending": False, "name": None, "backend": "none", "error": str(exc)}
    _PROBE = data
    if data.get("backend") in ("cuda", "rocm") and stamp:
        try:
            PROBE_CACHE.write_text(json.dumps({"stamp": stamp, "t": time.time(), "data": data}), encoding="utf-8")
        except OSError:
            pass


# ---------------------------------------------------------------- generation
def explain_failure(tail: list[str]) -> str:
    text = "\n".join(tail)
    low = text.lower()
    if "out of memory" in low or "cuda oom" in low or "hip out of memory" in low:
        return ("Out of GPU memory. Close games, video players and other GPU-heavy apps (also browser tabs with "
                "hardware acceleration), set a shorter Song length in Advanced and try again, or switch the device "
                "to CPU.")
    if "no module named" in low:
        return "Dependencies are missing. Run install.bat again."
    if "defaultcpuallocator" in low or "not enough memory" in low or "memoryerror" in low:
        return ("Not enough system RAM. Close other programs, increase the Windows page file, "
                "or use a shorter Song length.")
    if "cuda" in low and ("not available" in low or "no cuda" in low or "found no nvidia" in low):
        return "The GPU is not available to PyTorch. Update the driver and re-run install.bat, or choose the CPU device."
    if "hip error" in low or "hipblas" in low or "miopen" in low or "rocm" in low:
        return "AMD/ROCm error. In Advanced, try the eager kernel mode or the CPU device."
    if "connection" in low or "offline" in low or "couldn't connect" in low:
        return "Model files not found. Use the Download models button on the main page."
    return "Generation failed. See the log below for details.\n" + (tail[-1] if tail else "")


def start_generation(p: dict) -> dict:
    style = (p.get("style") or "").strip()
    lyrics = (p.get("lyrics") or "").strip()
    if not style:
        return {"error": "Please fill in the Style field."}
    if not lyrics:
        return {"error": "Please enter the lyrics (section tags like [Verse], [Chorus] are supported)."}
    cot = p.get("cot", "full")
    if cot not in ("full", "melody", "off"):
        cot = "full"
    abc = (p.get("abc") or "").strip() or None
    if abc and cot == "off":
        return {"error": "A custom ABC score requires the full or melody composition mode."}
    plan_only = p.get("stage") == "plan"
    if plan_only and cot == "off":
        return {"error": "Score preview needs the full or melody composition mode."}
    if len(lyrics) > 20000 or len(style) > 2000 or (abc and len(abc) > 200000):
        return {"error": "Input is too long."}
    if free_bytes(OUTPUTS) < 1.5 * 2**30:
        return {"error": "Less than 1.5 GB of free disk space. Free some space and try again."}

    info = _PROBE
    device = p.get("device", "auto")
    backend = info.get("backend")
    if device == "cpu" or (device == "auto" and backend not in ("cuda", "rocm")):
        device = "cpu"
    elif backend not in ("cuda", "rocm"):
        return {"error": "No GPU accelerator was found for PyTorch. Choose the CPU device or re-run install.bat."}

    ready = models_ready(force=True)
    local = ready["model"] and ready["vae"]

    try:
        seed = int(p.get("seed")) if p.get("seed") not in (None, "", "random") else random.randint(1, 2**31 - 1)
    except (TypeError, ValueError):
        seed = random.randint(1, 2**31 - 1)
    seed = abs(seed) % (2**31 - 1) or 1

    title = (p.get("title") or "").strip()[:60] or "Song"
    job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{slug(title)}"
    request = {"id": job_id, "style": style, "lyrics": lyrics, "cot": cot, "seed": seed}
    try:
        if p.get("cfg_scale") not in (None, ""):
            request["cfg_scale"] = float(p["cfg_scale"])
    except (TypeError, ValueError):
        pass
    if abc:
        request["abc"] = abc

    try:
        budget = float(p.get("budget") or 24)
    except (TypeError, ValueError):
        budget = 24.0
    gen_cfg: dict = {}
    try:
        tokens = int(p.get("max_tokens") or 9000)
    except (TypeError, ValueError):
        tokens = 9000
    tokens = max(300, min(9000, tokens))
    if tokens != 9000:
        gen_cfg["semantic"] = {"max_tokens": tokens}
    steps = {"fast": 16, "draft": 8}.get(p.get("quality", "standard"))
    if steps:
        gen_cfg["ode_steps"] = steps

    cmd = [python_exe(), "-u", "-m", CLI_MODULE, "generate", "--request", str(JOBS / f"{job_id}.json"),
           "--output", str(OUTPUTS), "--budget", str(budget)]
    if local:
        cmd.append("--offline")
    if gen_cfg:
        cmd += ["--config", str(JOBS / f"{job_id}.config.json")]
    if plan_only:
        cmd += ["--stage", "plan"]
    if device == "cpu":
        cmd += ["--device", "cpu"]
    lv = p.get("low_vram", "auto")
    if device != "cpu" and lv in ("on", "off"):
        cmd += ["--low-vram", lv]
    kernel = p.get("kernel", "auto")
    if kernel == "eager" or (kernel == "auto" and (device == "cpu" or backend != "cuda")):
        cmd += ["--backend", "torch-eager"]
    if p.get("fp8") and device != "cpu" and backend == "cuda" and lv == "off":
        cmd += ["--quantization", "fp8"]
    if p.get("offload_ar"):
        cmd.append("--offload-ar")

    job = {"id": job_id, "title": title, "status": "running", "started": time.time(), "style": style,
           "seed": seed, "cot": cot, "stage": "Starting…", "log_offset": 0,
           "kind": "plan" if plan_only else "song", "cancel": False}
    with S.lock:  # claim the queue atomically: two quick clicks cannot start two jobs
        if S.generating():
            return {"error": "A song is already being generated. Wait for it to finish or cancel it."}
        if S.downloading():
            return {"error": "Models are downloading. Wait until the download finishes."}
        if EDIT_LOCK.locked():
            return {"error": "An audio edit is running. Try again in a moment."}
        S.job, S.log, S.proc = job, [], None

    (JOBS / f"{job_id}.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    if gen_cfg:
        (JOBS / f"{job_id}.config.json").write_text(json.dumps(gen_cfg), encoding="utf-8")
    S.add_log("▶ " + " ".join(cmd[2:]))
    threading.Thread(target=_run_job, args=(job, cmd, plan_only, lyrics), daemon=True).start()
    return {"ok": True, "id": job_id}


def _run_job(job: dict, cmd: list[str], plan_only: bool, lyrics: str):
    job_id = job["id"]
    out = OUTPUTS / job_id
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=child_env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                bufsize=1, creationflags=CREATE_NO_WINDOW | BELOW_NORMAL)
    except Exception as exc:  # noqa: BLE001
        job.update(status="failed", error=f"Could not start Python: {exc}", finished=time.time(), stage="Failed")
        return
    with S.lock:
        S.proc = proc
    if job.get("cancel"):  # cancelled while starting
        kill_tree(proc)
    keep_awake(True)  # stop Windows from sleeping in the middle of a long generation
    try:
        for raw in proc.stdout:  # type: ignore[union-attr]
            line = raw.rstrip("\r\n")
            if line.strip():
                S.add_log(line[:500])
                job["stage"] = line.strip()[:160]
        code = proc.wait()
    finally:
        keep_awake(False)
    job["finished"] = time.time()
    expected = out / ("score.abc" if plan_only else "audio.flac")
    if job.get("cancel"):
        job["status"] = "cancelled"
        rmtree_retry(out)
    elif code == 0 and expected.exists():
        job["status"] = "done"
        meta = {"id": job_id, "title": job["title"], "style": job["style"], "lyrics": lyrics, "seed": job["seed"],
                "cot": job["cot"], "created": job["started"], "seconds": round(job["finished"] - job["started"]),
                "kind": job["kind"]}
        (out / "ui_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    else:
        job["status"] = "failed"
        with S.lock:
            tail = [x for x in S.log[-14:] if x.strip()]
        job["error"] = explain_failure(tail)
        try:  # keep diagnostics in app/_jobs, do not clutter the library folder
            (JOBS / f"{job_id}.log").write_text("\n".join(S.log), encoding="utf-8")
        except OSError:
            pass
        rmtree_retry(out)
    job["stage"] = {"done": "Done", "failed": "Failed", "cancelled": "Cancelled"}.get(job["status"], "")


def cancel_generation() -> dict:
    with S.lock:
        if S.generating() and S.job is not None:
            S.job["cancel"] = True
            kill_tree(S.proc)
            return {"ok": True}
    return {"error": "Nothing to cancel."}


# ---------------------------------------------------------------- model download
def start_download() -> dict:
    with S.lock:
        if S.downloading():
            return {"error": "A download is already running."}
        if S.generating():
            return {"error": "A song is being generated."}
        S.dl_log = []
        S.dl = {"bytes": 0, "total": 0, "speed": 0.0, "step": "Starting…", "error": None}
        proc = subprocess.Popen([python_exe(), "-u", str(ROOT / "app" / "download_models.py")], cwd=str(ROOT),
                                env=child_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1, creationflags=CREATE_NO_WINDOW)
        S.dl_proc = proc

    def pump():
        for raw in proc.stdout:  # type: ignore[union-attr]
            line = raw.rstrip("\r\n")
            try:
                if line.startswith("PROGRESS "):
                    _, b, t, sp = line.split()
                    S.dl.update(bytes=int(b), total=int(t), speed=float(sp))
                    continue
                if line.startswith("STEP "):
                    S.dl["step"] = line[5:]
                elif line.startswith("ERROR"):
                    S.dl["error"] = line[6:].strip() or "Download failed"
            except ValueError:
                pass
            S.dl_log.append(line[-300:])
            del S.dl_log[:-100]
        code = proc.wait()
        if code != 0 and not S.dl.get("error"):
            S.dl["error"] = "Download stopped. Press the button to resume where it left off."
        models_ready(force=True)

    threading.Thread(target=pump, daemon=True).start()
    return {"ok": True}


def cancel_download() -> dict:
    with S.lock:
        if S.downloading():
            S.dl["error"] = "Download cancelled. Press the button to resume."
            kill_tree(S.dl_proc)
            return {"ok": True}
    return {"error": "Nothing to cancel."}


# ---------------------------------------------------------------- library / files
def safe_out(job_id: str) -> Path | None:
    if not job_id or job_id.startswith("_") or not re.fullmatch(r"[\w\-]+", job_id, flags=re.UNICODE):
        return None
    path = (OUTPUTS / job_id).resolve()
    return path if OUTPUTS.resolve() in path.parents and path.is_dir() else None


def read_meta(d: Path) -> dict:
    try:
        return json.loads((d / "ui_meta.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_library() -> list[dict]:
    items = []
    for d in OUTPUTS.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        audio = d / "audio.flac"
        try:
            st = audio.stat()
        except OSError:
            continue
        meta = read_meta(d)
        meta.setdefault("id", d.name)
        meta.setdefault("title", d.name)
        meta.setdefault("created", st.st_mtime)
        meta["size_mb"] = round(st.st_size / 2**20, 1)
        meta["has_score"] = (d / "score.abc").exists()
        meta.pop("ops", None)
        items.append(meta)
    items.sort(key=lambda m: m.get("created", 0), reverse=True)
    return items


def find_upload(token: str) -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{12}", token or ""):
        return None
    return next((f for f in UPLOADS.glob(token + ".*") if f.suffix != ".part"), None)


def source_path(src: dict) -> tuple[Path | None, str]:
    if src.get("type") == "song":
        d = safe_out(src.get("id", ""))
        if not d:
            return None, "song"
        return d / "audio.flac", read_meta(d).get("title", d.name)
    path = find_upload(src.get("token", ""))
    return path, Path(src.get("name") or "song").stem


def run_edit(p: dict) -> dict:
    if S.generating() or S.downloading():
        return {"error": "Wait until the current generation or download finishes, then try again."}
    path, name = source_path(p.get("source") or {})
    if not path or not path.exists():
        return {"error": "Source song not found. Upload it again."}
    if not EDIT_LOCK.acquire(blocking=False):
        return {"error": "Another edit is still running."}
    out_dir = None
    try:
        import audio_tools as at
        if free_bytes(OUTPUTS) < 1 * 2**30:
            return {"error": "Less than 1 GB of free disk space."}
        started = time.time()
        data, sr = at.load_audio(path)
        out, notes = at.process(data, sr, p.get("ops") or {})
        del data
        title = (p.get("title") or "").strip()[:60] or f"{name} (edited)"
        job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{slug(title)}"
        out_dir = OUTPUTS / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        at.write_flac(out_dir / "audio.flac", out, sr)
        meta = {"id": job_id, "title": title, "created": time.time(), "kind": "edit",
                "seconds": round(time.time() - started), "source": name, "ops": p.get("ops") or {}}
        (out_dir / "ui_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "id": job_id, "notes": notes}
    except ValueError as exc:
        return {"error": str(exc)}
    except MemoryError:
        return {"error": "Not enough memory to edit this song. Close other programs or trim it first."}
    except Exception as exc:  # noqa: BLE001
        if out_dir is not None:
            rmtree_retry(out_dir, 2)
        return {"error": f"Editing failed: {exc}"}
    finally:
        EDIT_LOCK.release()


def ensure_export(d: Path, fmt: str) -> Path:
    """Return the path of audio.<fmt> inside song folder d, converting from FLAC when needed (atomic)."""
    target = d / f"audio.{fmt}"
    if fmt == "flac" or target.exists():
        return target
    import audio_tools as at
    with EDIT_LOCK:
        if not target.exists():
            at.convert(d / "audio.flac", target, fmt)
    return target


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "VitusSongs/1.1"
    timeout = 300  # drop stalled connections instead of leaking threads

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    # -- guards
    def allowed(self) -> bool:
        """Reject requests from other websites (DNS rebinding / cross-site POST) and foreign hosts."""
        host = self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]").lower()
        if host not in ALLOWED_HOSTS:
            return False
        origin = self.headers.get("Origin")
        if origin and (urlparse(origin).hostname or "") not in ALLOWED_HOSTS:
            return False
        return self.headers.get("Sec-Fetch-Site", "same-origin") in ("same-origin", "none", "same-site")

    # -- utilities
    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def json_body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n or n > 2 * 2**20:
            return {}
        try:
            value = json.loads(self.rfile.read(n).decode("utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def send_file(self, path: Path, download_name: str | None = None, cache: bool = False):
        try:
            size = path.stat().st_size
        except OSError:
            return self.send_error(HTTPStatus.NOT_FOUND)
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and size:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    return self.end_headers()
                status = 206
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download_name:
            ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", download_name) or "song"
            self.send_header("Content-Disposition",
                             f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(download_name)}")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass

    # -- GET / HEAD
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.allowed():
            return self.send_error(HTTPStatus.FORBIDDEN)
        url = urlparse(self.path)
        path, query = unquote(url.path), parse_qs(url.query)
        if path in ("/", "/index.html"):
            return self.send_file(STATIC / "index.html")
        if path.startswith("/static/"):
            f = (STATIC / path[len("/static/"):]).resolve()
            if STATIC.resolve() in f.parents and f.is_file():
                return self.send_file(f, cache=True)
            return self.send_error(HTTPStatus.NOT_FOUND)
        if path == "/api/ping":
            LAST["ping"] = time.time()
            return self.send_json({"app": "VitusSongs"})
        if path == "/api/status":
            with S.lock:
                job = dict(S.job) if S.job else None
                dl = dict(S.dl)
            return self.send_json({"models": models_ready(), "gpu": _PROBE, "job": job, "downloading": S.downloading(),
                                   "dl": dl, "editing": EDIT_LOCK.locked(), "now": time.time(),
                                   "free_gb": round(free_bytes(OUTPUTS) / 2**30, 1)})
        if path == "/api/log":
            try:
                since = int(query.get("since", ["0"])[0])
            except ValueError:
                since = 0
            with S.lock:
                off = S.job.get("log_offset", 0) if S.job else 0
                return self.send_json({"lines": S.log[max(0, since - off):], "next": off + len(S.log)})
        if path == "/api/library":
            return self.send_json(list_library())
        if path == "/api/peaks":
            src = {"type": "song", "id": query.get("id", [""])[0]} if "id" in query else \
                  {"type": "upload", "token": query.get("token", [""])[0]}
            audio, _ = source_path(src)
            if not audio or not audio.exists():
                return self.send_json({"error": "Not found"}, 404)
            try:
                import audio_tools as at
                return self.send_json(at.peaks(audio))
            except Exception as exc:  # noqa: BLE001
                return self.send_json({"error": str(exc)}, 500)
        m = re.fullmatch(r"/files/([\w\-]+)/(audio\.(?:flac|mp3|wav)|score\.abc)", path, flags=re.UNICODE)
        if m:
            d = safe_out(m.group(1))
            if not d:
                return self.send_error(HTTPStatus.NOT_FOUND)
            name = m.group(2)
            try:
                if name.startswith("audio."):
                    target = ensure_export(d, name[6:])
                else:
                    target = d / name
            except Exception as exc:  # noqa: BLE001
                return self.send_json({"error": f"Export failed: {exc}"}, 500)
            dl = None
            if "dl" in query:
                dl = slug(read_meta(d).get("title", d.name)) + Path(name).suffix
            return self.send_file(target, dl, cache=dl is None)
        self.send_error(HTTPStatus.NOT_FOUND)

    # -- POST
    def handle_upload(self, url):
        name = parse_qs(url.query).get("name", ["audio"])[0]
        ext = Path(name).suffix.lower()
        if ext not in AUDIO_EXT:
            return self.send_json({"error": "Unsupported file type. Use MP3, WAV, FLAC or OGG."}, 400)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD:
            self.close_connection = True
            return self.send_json({"error": "File is empty or larger than 600 MB."}, 413)
        if free_bytes(UPLOADS) < length * 2 + 2**28:
            self.close_connection = True
            return self.send_json({"error": "Not enough free disk space for this upload."}, 507)
        token = uuid.uuid4().hex[:12]
        part = UPLOADS / f"{token}{ext}.part"
        dest = UPLOADS / f"{token}{ext}"
        left = length
        try:
            with open(part, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            if left > 0:
                raise ConnectionError("Upload interrupted")
            import soundfile as sf
            info = sf.info(str(part))
            part.replace(dest)
        except ConnectionError:
            part.unlink(missing_ok=True)
            self.close_connection = True
            return
        except Exception:
            part.unlink(missing_ok=True)
            return self.send_json({"error": "This file could not be read as audio."}, 400)
        return self.send_json({"ok": True, "token": token, "name": name, "duration": round(info.duration, 2),
                               "samplerate": info.samplerate, "channels": info.channels})

    def do_POST(self):
        if not self.allowed():
            return self.send_error(HTTPStatus.FORBIDDEN)
        url = urlparse(self.path)
        path = url.path
        if path == "/api/upload":
            return self.handle_upload(url)
        data = self.json_body()

        def reply(res):
            self.send_json(res, 400 if "error" in res else 200)

        if path == "/api/generate":
            return reply(start_generation(data))
        if path == "/api/cancel":
            return reply(cancel_generation())
        if path == "/api/download-models":
            return reply(start_download())
        if path == "/api/cancel-download":
            return reply(cancel_download())
        if path == "/api/edit":
            return reply(run_edit(data))
        if path == "/api/bye":  # sent by the page when its window is closed or reloaded
            LAST["bye"] = time.time()
            return self.send_json({"ok": True})
        if path == "/api/prepare-export":
            d = safe_out(data.get("id", ""))
            fmt = data.get("fmt", "flac")
            if not d or fmt not in ("flac", "mp3", "wav"):
                return self.send_json({"error": "Not found"}, 404)
            try:
                target = ensure_export(d, fmt)
                return self.send_json({"ok": True, "size": target.stat().st_size,
                                       "url": f"/files/{quote(d.name)}/audio.{fmt}?dl=1"})
            except Exception as exc:  # noqa: BLE001
                return self.send_json({"error": f"Export failed: {exc}"}, 500)
        if path == "/api/open-folder":
            target = safe_out(data.get("id", "")) or OUTPUTS
            try:
                if os.name == "nt":
                    os.startfile(str(target))  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(target)])
                else:
                    subprocess.Popen(["xdg-open", str(target)])
            except Exception as exc:  # noqa: BLE001
                return self.send_json({"error": str(exc)}, 500)
            return self.send_json({"ok": True})
        if path == "/api/delete":
            d = safe_out(data.get("id", ""))
            if not d:
                return self.send_json({"error": "Not found"}, 404)
            rmtree_retry(d, 3)
            return self.send_json({"ok": True})
        if path == "/api/rename":
            d = safe_out(data.get("id", ""))
            title = (data.get("title") or "").strip()[:60]
            if not d or not title:
                return self.send_json({"error": "Invalid request"}, 400)
            meta = read_meta(d) or {"id": d.name}
            meta["title"] = title
            (d / "ui_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            return self.send_json({"ok": True})
        self.send_error(HTTPStatus.NOT_FOUND)


# ---------------------------------------------------------------- app window / lifecycle
def app_window(url: str):
    """Open VitusSongs in its own app-style window (Edge/Chrome --app), else the default browser."""
    candidates = []
    for var in ("ProgramFiles(x86)", "ProgramFiles", "LocalAppData"):
        base = os.environ.get(var)
        if base:
            candidates += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                           os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    for exe in candidates:
        if os.path.exists(exe):
            try:
                subprocess.Popen([exe, f"--app={url}", "--window-size=1320,900", "--no-first-run",
                                  "--no-default-browser-check", "--disable-extensions", "--disable-sync",
                                  f"--user-data-dir={ROOT / 'app' / '_profile'}"])
                return
            except OSError:
                pass
    webbrowser.open(url)


def already_running() -> bool:
    try:
        with urlopen(f"http://{HOST}:{PORT}/api/ping", timeout=1.5) as r:
            return json.loads(r.read().decode("utf-8")).get("app") == "VitusSongs"
    except Exception:
        return False


def watchdog(httpd):
    """Quit when the app window is gone and nothing is generating, downloading or editing."""
    while True:
        time.sleep(3)
        now = time.time()
        busy = S.generating() or S.downloading() or EDIT_LOCK.locked()
        closed = bool(LAST["bye"] and now - LAST["bye"] > 10 and LAST["ping"] < LAST["bye"])
        if (closed or now - LAST["ping"] > 180) and not busy:
            httpd.shutdown()
            return


def housekeeping():
    now = time.time()
    for folder, max_age in ((UPLOADS, 3 * 86400), (JOBS, 7 * 86400)):
        for f in folder.glob("*"):
            try:
                if f.is_file() and now - f.stat().st_mtime > max_age:
                    f.unlink()
            except OSError:
                pass
    log = ROOT / "app" / "server.log"
    try:
        if log.exists() and log.stat().st_size > 1_000_000:
            log.replace(log.with_suffix(".old"))
    except OSError:
        pass


def main():
    housekeeping()
    if sys.stdout is None:  # started without a console (VitusSongs.vbs / pythonw)
        sys.stdout = sys.stderr = open(ROOT / "app" / "server.log", "a", encoding="utf-8", buffering=1)
    interactive = os.environ.get("YUE_STUDIO_NO_BROWSER") != "1"
    if interactive and already_running():
        app_window(f"http://{HOST}:{PORT}/")
        return 0
    httpd = None
    port = PORT
    for _ in range(20):
        try:
            httpd = ThreadingHTTPServer((HOST, port), Handler)
            break
        except OSError:
            port += 1
    if httpd is None:
        print("Could not bind a port. Close other copies of VitusSongs.")
        return 1
    httpd.daemon_threads = True
    url = f"http://{HOST}:{port}/"
    print("=" * 60)
    print("  VitusSongs is running:", url)
    print("=" * 60)
    threading.Thread(target=run_probe, daemon=True).start()
    if interactive:
        threading.Thread(target=watchdog, args=(httpd,), daemon=True).start()
        threading.Timer(0.8, lambda: app_window(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with S.lock:
            kill_tree(S.proc if S.generating() else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
