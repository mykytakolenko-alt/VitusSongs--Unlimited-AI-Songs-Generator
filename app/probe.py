"""Detect which accelerator PyTorch sees. Prints one JSON line."""
import json
import os


def ram_gib():
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("pt", ctypes.c_ulonglong), ("pa", ctypes.c_ulonglong),
                            ("vt", ctypes.c_ulonglong), ("va", ctypes.c_ulonglong), ("ve", ctypes.c_ulonglong)]
            m = MS(); m.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.total / 2**30, 1)
        with open("/proc/meminfo") as f:
            return round(int(f.readline().split()[1]) / 2**20, 1)
    except Exception:
        return None


out = {"pending": False, "name": None, "backend": "none", "ram_gib": ram_gib(), "cpu_threads": os.cpu_count()}
try:
    import torch
    out["torch"] = torch.__version__
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        out.update(name=props.name, total_gib=round(props.total_memory / 2**30, 1),
                   backend="rocm" if torch.version.hip else "cuda")
    else:
        out["backend"] = "cpu"
        out["name"] = "CPU"
except Exception as exc:  # noqa: BLE001
    out["error"] = str(exc)
print(json.dumps(out, ensure_ascii=False))
