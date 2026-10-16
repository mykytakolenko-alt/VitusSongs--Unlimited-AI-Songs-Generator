# 🎵 VitusSongs

**Generate unlimited AI songs locally on your Windows PC - no subscriptions, no cloud, no limits.**

[![![Version](https://img.shields.io/badge/version-0.1.6-blue)](https://github.com/mykytakolenko-alt/VitusSongs--Unlimited-AI-Songs-Generator)
[![Code License](https://img.shields.io/badge/Code%20License-Apache--2.0-green)](LICENSE)
[![Model License](https://img.shields.io/badge/Model%20Weights-CC%20BY--NC%204.0-orange)](MODEL_LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-yellow)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey)](https://www.microsoft.com/windows)

---

## 📌 Overview

**VitusSongs** is a Unlimited AI songs generator that works perfectrly on you 11/10 Windows PC with 6-8 GB VRAM and at least 16 GB RAM.

You can generate unlimited songs locally and also choose your style, lyrics/no lyrics, melody and also access advanced setings to make generations more accurate.

Works with Nvidia/AMD Graphic cards.

Download .exe https://github.com/mykytakolenko-alt/VitusSongs--Unlimited-AI-Songs-Generator/releases/tag/v0.1.6

---

## ✨ Features

- Polished & Light Web UI — Optimized interface with no heavy blur effects; stays snappy on weak PCs and iGPUs.
- NVIDIA CUDA Path — Native high-speed generation path for NVIDIA GPUs.
- AMD ROCm for Windows — Experimental support via ROCm (RX 7000/9000 series and ROCm nightly `gfx103X` builds for RX 6600+).
- Smart Low-VRAM Mode — Run generation on 8 GB GPUs (e.g., RX 6600, RTX 3060) by automatically splitting the model into an AR half (~4.3 GB) and NAR half (~2.8 GB), streaming weights host-to-GPU on the fly without extra RAM re-allocations.
- Full DSP Song Modifier — Upload MP3/WAV/FLAC/OGG or edit generated tracks: pitch-preserving tempo shift, pitch correction, low/high-cut filters, EQ, volume, fades, trim, normalize, reverse, and mono conversion.
- AI Music & Score Editing — Load generated ABC score notation and lyrics to tweak notes or BPM, then re-render a brand-new recording with YuE2.
- 100% Private & Safe Local Server — Binds strictly to `127.0.0.1/localhost` with Cross-Site Request Forgery (CSRF) protection. Auto-cleans temp files and incomplete generations.
- Export Options — Download songs in FLAC, WAV, or high-quality VBR MP3 formats.

---

## 🚀 Installation & Usage

### 1. Installation
-Download **`VitusSongs-Setup-0.1.6.exe`** from the (https://github.com/mykytakolenko-alt/VitusSongs--Unlimited-AI-Songs-Generator/releases/tag/v0.1.6) page.
-Run the `.exe` file from releases and follow the setup wizard.
-Launch the app and download the required models (~8–10 GB) directly inside the user interface.

> **Manual Installation**:
1. Unzip the project release into a path **WITHOUT spaces or non-English characters** (e.g., `C:\VitusSongs`). Delete any old `.venv` folders if upgrading.
2. Run `install.bat`.
   - Automatically detects or installs Python 3.12 via `winget`.
   - Identifies your GPU architecture and installs matching CUDA or ROCm PyTorch builds.
   - Creates a **VitusSongs** shortcut on your Desktop.
3. Download models inside the UI or via `download_models.bat` (~8–10 GB download; supports resume).

> **Manual Target Override**: You can pass custom GPU targets directly:
> - `install.bat cuda`
> - `install.bat amd gfx1032` *(RX 6600/6650)* | `gfx1030` *(RX 6800/6900)* | `gfx1100` *(RX 7900)* | `gfx1201` *(RX 9070)*
> - `install.bat cpu`

### 2. Running the App

- Double-click the **VitusSongs** desktop shortcut (or run `VitusSongs.vbs`).
- To launch with a diagnostic terminal console, run `start.bat`.
- To diagnose GPU/PyTorch detection issues, execute `check_system.bat` or `diagnose.bat`.

### 3. Basic Workflow

1. **Style**: Select language, genre, instruments, vocals, and tempo presets.
2. **Lyrics**: Input text formatted with tags like `[Verse]`, `[Chorus]`, `[Bridge]`.
3. **Generate**: Click **Create song**. Tracks will appear in **My Library**.
4. **Edit & Modify**: Trim, shift pitch, tweak EQ, or edit the ABC score to re-render fresh AI variations.

## 📊 Feature Comparison

| Feature | Official YuE2 | VitusSongs |
|---|:---:|:---:|
| **Supported OS** | Linux only | **Windows 10 / 11** |
| **Interface** | CLI / Jupyter | **Web UI (Edge App Mode)** |
| **AMD GPU Support** | ❌ | ✅ **ROCm (Experimental)** |
| **8 GB VRAM Support** | ❌ (OOM) | ✅ **Low-VRAM Mode** |
| **Setup Process** | Manual | **One-click `install.bat`** |
| **Audio DSP Modifier** | ❌ | ✅ **Built-in** |
| **Score Editing** | ❌ | ✅ **ABC Re-render** |

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 (64-bit) | Windows 11 (64-bit) |
| **RAM** | 16 GB | 32 GB |
| **VRAM** | 8 GB (Low-VRAM mode) | 12 GB+ |
| **Disk Space** | ~25 GB free | 50 GB SSD |
| **NVIDIA GPU** | RTX series (CUDA) | RTX 3070 / 4070+ |
| **AMD GPU** | RX 6600 / 6700 (`gfx103X`) | RX 7000 / 9000 (`gfx110X` / `gfx120X`) |
| **CPU Fallback** | Any 64-bit CPU | 8+ physical cores (e.g., Ryzen 7) |

> ⚠️ **Note on AMD / Older GPUs**: AMD Radeon VII / Vega GPUs have no ROCm support on Windows and run in CPU mode only. CPU-only mode requires 16–32 GB RAM (generation takes several hours per track).

---

## ⚙️ Performance Tuning for Weaker PCs

- **Song Length Cap**: Default on weak GPUs is 4500 tokens (~3 minutes). Lowering this value accelerates generation and reduces VRAM/RAM overhead.
- **ODE Quality Steps**: Reduce last-stage ODE steps (16 steps for faster generation, 8 steps for draft mode).
- **Background Execution**: VitusSongs runs at lower process priority and keeps Windows awake during render jobs, keeping your desktop responsive.
- **Physical Cores**: Set CPU math threads to physical core counts (e.g., 8 on Ryzen 7) for optimal CPU execution.
- **Windows Pagefile**: If running on 16 GB RAM, configure a large Windows page file (*System > Advanced > Performance*).

---


---

## 🛠️ Technical Implementation & Patches

This project adapts the upstream YuE2 repository (which officially targets Linux + NVIDIA) to Windows via specialized runtime patches:
- **Portable SDPA Execution**: Replaces Linux-bound `flash-attention` / `cuDNN` dependencies with portable Scaled Dot-Product Attention (SDPA).
- **AMD Compatibility Patches**: Removed CUDA BF16 hardware enforcement checks (`cuda_graph.py`, `pipeline.py`).
- **Low-VRAM Weight Offloading**: Custom weight swapping implementation (`src/yue2/lowvram.py`) pinned in CPU memory, enabling low-latency Host-to-GPU transfers during execution steps.

---

## ⚖️ Licensing & Attributions

- **Source Code**: Distributed under the [Apache 2.0 License](LICENSE).
- **Model Weights**: Subject to [CC BY-NC 4.0](MODEL_LICENSE) (Non-commercial use).
- **Third-Party Libraries**: Refer to [THIRD_PARTY_NOTICES.MD](THIRD_PARTY_NOTICES.MD) and the `licenses/` folder for ups
