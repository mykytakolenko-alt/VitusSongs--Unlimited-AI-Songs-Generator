"""Audio editing helpers for the VitusSongs song modifier (numpy + soundfile only).

Everything here is classic DSP: it changes an existing recording (trim, tempo, pitch, EQ,
loudness, fades). It does not re-compose the music - that is what the score editor is for.
"""
from __future__ import annotations

import numpy as np

MP3_RATES = (48000, 44100, 32000, 24000, 22050, 16000)


def load_audio(path):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, int(sr)


# ----------------------------------------------------------------------------- time stretch
def _window(n_fft):
    return np.hanning(n_fft + 1)[:-1].astype(np.float32)


def _stft(x, n_fft, hop, block=2048):
    pad = n_fft // 2
    view = np.lib.stride_tricks.sliding_window_view(np.pad(x, (pad, pad), mode="reflect"), n_fft)[::hop]
    window = _window(n_fft)
    spec = np.empty((n_fft // 2 + 1, view.shape[0]), dtype=np.complex64)
    for a in range(0, view.shape[0], block):  # block-wise: no giant temporary arrays
        chunk = view[a:a + block] * window
        spec[:, a:a + len(chunk)] = np.fft.rfft(chunk, axis=1).T
    return spec


def _istft(spec, n_fft, hop, length, block=2048):
    window = _window(n_fft)
    n_frames = spec.shape[1]
    out = np.zeros(hop * (n_frames - 1) + n_fft, dtype=np.float32)
    norm = np.zeros_like(out)
    w2 = window * window
    for a in range(0, n_frames, block):
        frames = np.fft.irfft(spec[:, a:a + block].T, n=n_fft, axis=1).astype(np.float32) * window
        for i in range(frames.shape[0]):
            pos = (a + i) * hop
            out[pos:pos + n_fft] += frames[i]
            norm[pos:pos + n_fft] += w2
    out /= np.maximum(norm, 0.1 * float(norm.max()))  # floor: no blow-up at the very edges
    out = out[n_fft // 2:]
    if len(out) < length:
        out = np.pad(out, (0, length - len(out)))
    return out[:length]


def time_stretch(x, rate, n_fft=2048, hop=512):
    """Pitch-preserving tempo change of a mono signal. rate > 1 = faster / shorter."""
    if abs(rate - 1.0) < 1e-4:
        return x
    spec = _stft(x, n_fft, hop)
    bins, n_frames = spec.shape
    steps = np.arange(0, n_frames - 1, rate)
    expected = (hop * np.linspace(0, np.pi, bins)).astype(np.float32)
    phase = np.angle(spec[:, 0]).astype(np.float32)
    mag = np.abs(spec)
    ang = np.angle(spec).astype(np.float32)
    del spec  # keep peak memory low on weak PCs
    out = np.empty((bins, len(steps)), dtype=np.complex64)
    for t, step in enumerate(steps):
        i, a = int(step), step - int(step)
        m = (1 - a) * mag[:, i] + a * mag[:, i + 1]
        out[:, t] = m * np.exp(1j * phase)
        d = ang[:, i + 1] - ang[:, i] - expected
        d -= 2 * np.pi * np.round(d / (2 * np.pi))
        phase = phase + expected + d
    return _istft(out, n_fft, hop, int(round(len(x) / rate)))


def resample_speed(x, factor, block=1 << 20):
    """Play back `factor` times faster (changes length and pitch), Catmull-Rom interpolation."""
    n_out = int(len(x) / factor)
    out = np.empty(n_out, dtype=np.float32)
    last = len(x) - 1
    for a in range(0, n_out, block):
        pos = np.arange(a, min(a + block, n_out), dtype=np.float64) * factor
        i = np.floor(pos).astype(np.int64)
        t = (pos - i).astype(np.float32)
        p0, p1 = x[np.clip(i - 1, 0, last)], x[np.clip(i, 0, last)]
        p2, p3 = x[np.clip(i + 1, 0, last)], x[np.clip(i + 2, 0, last)]
        out[a:a + len(t)] = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                                   + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)
    return out


def pitch_shift(x, semitones):
    if abs(semitones) < 1e-3:
        return x
    factor = 2.0 ** (semitones / 12.0)
    return resample_speed(time_stretch(x, 1.0 / factor), factor)


# ----------------------------------------------------------------------------- EQ
def equalize(x, sr, bass_db=0.0, treble_db=0.0, lowcut=0.0, highcut=0.0):
    if not (bass_db or treble_db or lowcut or highcut):
        return x
    n = len(x)
    nfft = 1 << max(10, (n - 1).bit_length())  # power of two: pocketfft is slow on awkward sizes
    spec = np.fft.rfft(x, n=nfft)
    f = np.maximum(np.fft.rfftfreq(nfft, 1.0 / sr), 1e-3).astype(np.float32)
    gain_db = np.zeros_like(f)
    if bass_db:
        gain_db += bass_db / (1.0 + (f / 200.0) ** 4)
    if treble_db:
        gain_db += treble_db / (1.0 + (4000.0 / f) ** 4)
    gain = 10.0 ** (gain_db / 20.0)
    if lowcut:
        gain = gain / np.sqrt(1.0 + (lowcut / f) ** 4)
    if highcut:
        gain = gain / np.sqrt(1.0 + (f / highcut) ** 4)
    spec *= gain.astype(np.float32)
    return np.fft.irfft(spec, n=nfft)[:n].astype(np.float32)


# ----------------------------------------------------------------------------- pipeline
def _num(ops, key, default=0.0):
    try:
        value = ops.get(key)
        return default if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return default


def process(data, sr, ops):
    """data: float32 [samples, channels]. Returns (data, notes)."""
    notes = []
    start, end = _num(ops, "trim_start"), _num(ops, "trim_end")
    a = max(0, int(start * sr))
    b = int(end * sr) if end and end > start else data.shape[0]
    data = data[a:min(b, data.shape[0])]
    if data.shape[0] < sr // 10:
        raise ValueError("The selected part is too short.")
    if ops.get("reverse"):
        data = data[::-1]
    tempo = min(2.0, max(0.5, _num(ops, "tempo", 1.0)))
    pitch = min(12.0, max(-12.0, _num(ops, "pitch", 0.0)))
    bass, treble = _num(ops, "bass_db"), _num(ops, "treble_db")
    lowcut, highcut = _num(ops, "lowcut"), _num(ops, "highcut")
    chans = []
    for c in range(data.shape[1]):
        x = np.ascontiguousarray(data[:, c])
        x = time_stretch(x, tempo)
        x = pitch_shift(x, pitch)
        x = equalize(x, sr, bass, treble, lowcut, highcut)
        chans.append(x)
    n = min(len(c) for c in chans)
    data = np.stack([c[:n] for c in chans], axis=1)
    if ops.get("mono") and data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    gain = _num(ops, "gain_db")
    if gain:
        data = data * (10.0 ** (gain / 20.0))
    if ops.get("normalize"):
        peak = float(np.max(np.abs(data))) or 1.0
        data = data * (10.0 ** (-1.0 / 20.0) / peak)
    for key, edge in (("fade_in", "in"), ("fade_out", "out")):
        length = min(int(_num(ops, key) * sr), data.shape[0])
        if length > 1:
            ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, length, dtype=np.float32)))[:, None]
            if edge == "in":
                data[:length] *= ramp
            else:
                data[-length:] *= ramp[::-1]
    peak = float(np.max(np.abs(data)))
    if peak > 0.999:
        data = data * (0.999 / peak)
        notes.append("Level reduced slightly to avoid clipping.")
    return data.astype(np.float32), notes


# ----------------------------------------------------------------------------- export
def write_flac(path, data, sr):
    import soundfile as sf
    sf.write(str(path), data, sr, subtype="PCM_24", format="FLAC")


def _writer(path, fmt, sr, channels):
    import soundfile as sf
    if fmt == "wav":
        return sf.SoundFile(str(path), "w", samplerate=sr, channels=channels, format="WAV", subtype="PCM_24")
    try:
        return sf.SoundFile(str(path), "w", samplerate=sr, channels=channels, format="MP3",
                            subtype="MPEG_LAYER_III", bitrate_mode="VARIABLE", compression_level=0.1)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"MP3 encoding is not available in this soundfile build: {exc}") from exc


def convert(src, dst, fmt):
    """Convert FLAC/WAV to 'mp3' or 'wav' by streaming blocks (low memory), written atomically."""
    import os
    import soundfile as sf
    from pathlib import Path
    if fmt not in ("mp3", "wav"):
        raise ValueError("Unsupported format")
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".part")
    try:
        with sf.SoundFile(str(src)) as reader:
            sr, channels = reader.samplerate, min(reader.channels, 2 if fmt == "mp3" else reader.channels)
            if fmt == "mp3" and sr not in MP3_RATES:  # rare: resample the whole file once
                data = reader.read(dtype="float32", always_2d=True)[:, :channels]
                data = np.stack([resample_speed(np.ascontiguousarray(data[:, c]), sr / 48000)
                                 for c in range(channels)], axis=1)
                with _writer(tmp, fmt, 48000, channels) as writer:
                    writer.write(np.clip(data, -1.0, 1.0))
            else:
                with _writer(tmp, fmt, sr, channels) as writer:
                    for block in reader.blocks(blocksize=1 << 18, dtype="float32", always_2d=True):
                        writer.write(np.clip(block[:, :channels], -1.0, 1.0))
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def peaks(path, n=600):
    """Waveform overview for the UI, computed by streaming (works for very large files)."""
    import soundfile as sf
    with sf.SoundFile(str(path)) as f:
        frames, sr = len(f), f.samplerate
        per = max(1, frames // n)
        out = np.zeros(n, dtype=np.float32)
        index = 0
        while index < n:
            block = f.read(per * 64, dtype="float32", always_2d=True)
            if not len(block):
                break
            m = np.abs(block).max(axis=1)
            usable = (len(m) // per) * per
            if usable:
                bins = m[:usable].reshape(-1, per).max(axis=1)
                take = min(len(bins), n - index)
                out[index:index + take] = bins[:take]
                index += take
            if usable < len(m) and index < n:  # tail shorter than one bin
                out[index] = m[usable:].max()
                index += 1
        top = float(out.max()) or 1.0
        return {"peaks": [round(float(v) / top, 3) for v in out], "duration": round(frames / sr, 2), "samplerate": sr}
