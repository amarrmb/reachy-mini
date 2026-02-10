# Baskd × Reachy Mini — Built on NVIDIA

> **For GTC booth conversations. One slide, one message.**

---

## Built on NVIDIA. Runs Without It.

| Layer | NVIDIA Technology | What It Does |
|-------|-------------------|-------------|
| **Hardware** | Jetson Thor (128GB unified) | Edge compute — all AI on-device |
| **Speech-to-Text** | Nemotron 0.6B | 24ms voice recognition, CUDA-accelerated |
| **Language Model** | vLLM on NVIDIA GPU | Qwen2.5-VL-7B at NVFP4 — 345ms text, 643ms vision |
| **Quantization** | NVFP4 (FP4) | 7B model in 6.8 GiB — 5x faster vision than BF16 |
| **Runtime** | CUDA 13.0 + FLASH_ATTN | Native Blackwell SM110 acceleration |

**Total VRAM: ~8 GiB of 128 GiB.** The rest is available for your workloads.

---

## How We're Different from the CES Demo

|  | NVIDIA/Brev (CES 2026) | Baskd (This Demo) |
|--|------------------------|-------------------|
| **Speech** | ElevenLabs cloud STT + TTS | Nemotron STT + Kokoro TTS — **100% local** |
| **Languages** | English only | **9 languages**, near-human quality |
| **Cameras** | Browser webcam | USB + RTSP + phone (WebRTC) |
| **Tools** | Wikipedia search | **20+ built-in + extensible plugins** |
| **Proactive Vision** | None | Watches for conditions, speaks unprompted |
| **VRAM** | ~93 GiB (DGX Spark) | **~8 GiB** (Jetson Thor) |
| **Network** | Required (ElevenLabs) | **Zero dependency — works in airplane mode** |
| **Hardware** | DGX Spark ($3,999) | Jetson Thor (~$2,000 edge device) |

---

## Complementary to NVIDIA Ecosystem

We don't compete with NVIDIA — we build on it.

- **NVIDIA provides**: Jetson hardware, Nemotron models, vLLM serving, CUDA runtime
- **Baskd provides**: The operational layer — voice assistant, robot control, tool system, fleet management

**Same relationship as Android and Google Play Services.**
The edge runtime is open. The cloud services (training, fleet management, improvement loop) are commercial.

---

## One-Liner for Booth Visitors

> *"Same Reachy Mini as the CES keynote. 12x less memory. 9 languages. Zero cloud. And you can add a new capability in 10 lines of Python."*

---

## Technical Specs

```
Pipeline: Mic → Nemotron STT (24ms) → vLLM (345ms) → Kokoro TTS (80ms) → Speaker
Latency: ~700ms wake-to-speech
Motion:  50Hz MotionManager (breathing + audio-reactive sway + listening)
Tools:   20+ built-in + external plugin system
Vision:  Multi-camera VLM with proactive monitoring
Network: NOT REQUIRED
```

---

## Contact

**Baskd** — Physical AI Infrastructure
- Web: baskd.io
- Demo: Ask Reachy anything. Or tell it to dance.
