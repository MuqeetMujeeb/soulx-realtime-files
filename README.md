# FlashHead × Gemini — Live Speech-to-Speech Avatar

A real-time conversational avatar. You speak into your browser microphone, Google
Gemini (Live API) responds with speech, and the SoulX-FlashHead talking-head model
animates a single portrait photo to lip-sync that speech — streamed back to your
browser as live video.

```
your mic ──► Gemini Live ──► 24kHz speech ──► resample 16kHz ──► FlashHead (talking head)
                                                                        │
   browser ◄──── WebSocket (video + audio) ◄──────────────────────────┘
```

This repo has three pieces:

| File | Role |
|------|------|
| `setup_soulx.sh` | One-shot installer for a fresh GPU pod (deps, models, launch helper) |
| `flashhead_gemini_server.py` | The server: Gemini bridge + FlashHead pipeline + WebSocket streaming |
| `client.html` | The browser UI: mic capture, video playback, latency metrics |

---

## Quick start

On a fresh GPU pod (RTX 3090 Ti / 4090 class, CUDA 12.8, Python 3.11):

```bash
# 1. run the installer (clones repo, installs everything, downloads models)
bash setup_soulx.sh

# 2. drop your two files into the project
#    flashhead_gemini_server.py  and  client.html
#    into  /workspace/SoulX-FlashHead/

# 3. put your portrait at examples/prototype.png (or edit COND_IMAGE)

# 4. set your key and launch
export GEMINI_API_KEY='your_key_here'
./run_server.sh
```

Then open the server's port (`7860`) in your browser, click **Start**, allow the
microphone, and talk.

> **Hardware note:** use an Ampere (sm_86, e.g. 3090 Ti) or Ada (sm_89, e.g. 4090)
> GPU. These have a mature software path (prebuilt flash-attn wheel + SageAttention).
> Blackwell (sm_120, e.g. 5090) requires bleeding-edge builds and is not recommended.

---

## 1. `setup_soulx.sh`

An **idempotent** installer (safe to re-run; it skips steps already done). Targets
**CUDA 12.8 + Python 3.11**. Run it once per fresh pod.

### What it does, in order

1. **Checks the environment** — warns if `nvcc` isn't CUDA 12.8.
2. **Installs system packages** (ffmpeg, build tools, etc.).
3. **Clones** SoulX-FlashHead from `github.com/Soul-AILab/SoulX-FlashHead`.
4. **Pins torch** `2.7.1 + cu128` with `torchvision 0.22.1` (matches xformers 0.0.31
   and the flash-attn wheel). If the base image ships a different torch, this
   replaces it.
5. **Installs FlashHead requirements** (filters out the `nvidia-nccl-cu12` pin; auto-
   recovers from the common `blinker` distutils error by retrying with
   `--ignore-installed blinker`).
6. **Installs flash-attn** from the prebuilt wheel
   `flash_attn-2.8.0.post2+cu12torch2.7...cp311...whl` (skips the ~30-min source build).
7. **Compiles SageAttention** from source for your GPU arch (the speed unlock).
8. **Downloads models** (~15GB total): `Soul-AILab/SoulX-FlashHead-1_3B` and
   `facebook/wav2vec2-base-960h` into `./models/`.
9. **Writes `run_server.sh`** — a launch helper that sets the torch.compile cache
   dir, `LD_LIBRARY_PATH`, checks for `GEMINI_API_KEY`, and starts the server.
10. **Verifies** the installed stack (prints versions of torch, flash_attn,
    sageattention, diffusers, transformers, cv2, etc.).

### Known fix you may need

The bundled FlashHead `requirements.txt` pins `mediapipe==0.10.9`, which has been
removed from PyPI. If the requirements install fails on mediapipe, relax the pin:

```bash
cd /workspace/SoulX-FlashHead
grep -v "nvidia-nccl-cu12" requirements.txt \
  | sed 's/mediapipe==0.10.9/mediapipe==0.10.14/' > requirements_fixed.txt
pip install --ignore-installed blinker
pip install -r requirements_fixed.txt
# then re-run setup_soulx.sh — it resumes from where it left off
```

### Notes

- **Packages live in the container**, not in `/workspace`. On most pod providers the
  container is wiped on terminate, so re-run `setup_soulx.sh` on each fresh pod.
  `/workspace` (models, code, compile cache) persists across stop/start.
- First server launch triggers a one-time **torch.compile (~4–5 min)** during the
  silence-priming step. This is normal; wait for `Server ready.`

---

## 2. `flashhead_gemini_server.py`

The core server. FastAPI + WebSocket. Loads the FlashHead pipeline once at startup,
bridges Gemini Live audio into it, and streams the resulting video/audio to the
browser.

### Run it

```bash
export GEMINI_API_KEY='your_key_here'
./run_server.sh          # or: python flashhead_gemini_server.py
```

Binds `0.0.0.0:7860`. Serves `client.html` at `/` and the WebSocket at `/ws`.

### Configuration (top of the file)

**Model / input**
| Setting | Default | Meaning |
|---------|---------|---------|
| `CKPT_DIR` | `models/SoulX-FlashHead-1_3B` | FlashHead checkpoint |
| `WAV2VEC_DIR` | `models/wav2vec2-base-960h` | Audio encoder |
| `MODEL_TYPE` | `lite` | `lite` (single-GPU, 4-step) or `pro` (dual-GPU) |
| `COND_IMAGE` | `examples/prototype.png` | The portrait the avatar animates |
| `BASE_SEED` | `9999` | Diffusion seed (motion variation; no speed effect) |
| `USE_FACE_CROP` | `True` | Crop to the detected face before generating |

**Gemini**
| Setting | Default | Meaning |
|---------|---------|---------|
| `GEMINI_MODEL` | `gemini-3.1-flash-live-preview` | Live API model |
| `GEMINI_SR` | `24000` | Gemini output sample rate (resampled to 16k for FlashHead) |
| `SYSTEM_INSTRUCTION` | (1–3 sentence persona) | Keeps turns short for low latency |

**Video delivery mode** — the big toggle
| Setting | Default | Meaning |
|---------|---------|---------|
| `VIDEO_MODE` | `"frames"` | `"frames"` = stream JPEGs (low latency); `"clips"` = batch into MP4s (smooth, higher latency) |
| `CLIP_CHUNKS` | `2` | clips mode only: chunks per MP4 (1 ≈ 1.1s ... bigger = smoother but laggier) |

**Pacing / idle**
| Setting | Default | Meaning |
|---------|---------|---------|
| `PREBUFFER_FRAMES` | `8` | Frames buffered before playback starts |
| `ENABLE_IDLE` | `True` | Feed low-noise during silence so the face doesn't freeze |
| `IDLE_GAP_SEC` | `0.4` | Silence before idle motion kicks in |
| `IDLE_QUEUE_MAX` | `6` | Only generate idle frames when the queue is below this (keeps idle off a speech backlog) |

**Body compositing** (experimental — head-on-static-body)
| Setting | Default | Meaning |
|---------|---------|---------|
| `ENABLE_COMPOSITE` | `False` | Master on/off |
| `BODY_IMAGE` | `examples/body.png` | Static body photo to paste the animated head onto |
| `COMPOSITE_X/Y` | `0` | Head position offset (px) |
| `COMPOSITE_SCALE` | `1.0` | Scale the head before pasting |
| `FEATHER_PX` | `25` | Soften the mask edge (seam blend) |

> If your `COND_IMAGE` already includes the body/shoulders, leave
> `ENABLE_COMPOSITE = False` — FlashHead generates the whole frame seamlessly and
> no compositing is needed. Compositing is only for pasting an animated face from
> one image onto a *separate* body image.

### The two video modes

**`frames` mode (default — low latency).** Each generated frame is JPEG-encoded and
streamed individually. The browser jitter-buffers and paints at a steady 25fps. Best
for live back-and-forth conversation (sub-second response). The browser does the
smoothing.

**`clips` mode (smooth — higher latency).** `CLIP_CHUNKS` chunks of frames are batched
into a short MP4 (audio muxed in), and the browser plays them with the native video
player. Very smooth and perfectly A/V-synced, but adds a `CLIP_CHUNKS × ~1.1s` latency
floor (it must wait for a full clip). Best for presentations/kiosks where a few
seconds of startup is acceptable.

### What loads in the pipeline

DiT 1.3B (the talking-head diffusion transformer, 4 distilled steps for Lite) +
LTX-VAE (decodes latents → pixels) + Wav2Vec2 (audio → embeddings) + MediaPipe
(face detection) + your encoded reference photo. ~14GB resident on the GPU.

### Logs to watch

- `[latency] chunk gen: Xms for N frames (Y fps)` — generation speed (distance-independent).
- `[feeder] +N samples, fired K chunk(s) ... q=Q` — audio fed, chunks generated, queue depth.
- `[stats] mic=... frames_sent=... frame_q=... chunks=... idle=...` — periodic counters.
- `turn ended; listening for next turn...` — confirms multi-turn conversation works.
- `[clip] emitted MP4 ...KB` — clips mode only.

---

## 3. `client.html`

The browser front end. Single self-contained file — no build step. Served at `/`.

### What it does

- **Captures the microphone**, downsamples to 16kHz int16 PCM, sends it as binary
  over the WebSocket.
- **Receives** tagged messages and routes by the first byte:
  - `0x01` = JPEG video frame (frames mode)
  - `0x02` = 24kHz PCM audio (frames mode; Gemini's voice)
  - `0x03` = MP4 clip (clips mode)
  - text = latency ping echo
- **Video jitter buffer (frames mode):** queues incoming frames and paints one every
  40ms via `requestAnimationFrame` — steady 25fps regardless of bursty arrival. This
  is the fix for "bad internet"-looking stutter.
- **Double-buffered clip player (clips mode):** two `<video>` elements; one plays
  while the other preloads the next clip, so transitions are gap-free.
- **Queued audio playback:** schedules Gemini's audio chunks back-to-back.
- **Latency ping:** sends a timestamp every second; the server echoes it; the round
  trip is shown as `net XXms` (your true network latency to the pod).

### The metrics line

Under the video you'll see:

```
recv 329  shown 277  buffer 52  net 371ms   [clips N(qM) in clips mode]
```

- `recv` / `shown` — frames received vs painted (both should climb steadily)
- `buffer` — jitter-buffer depth (frames mode)
- `net` — round-trip network latency to the pod (the GPU-distance penalty; lower is
  snappier — favor a pod geographically near you)
- `clips N(qM)` — clips received and queued (clips mode only)

### Tuning (top of the `<script>`)

| Setting | Default | Meaning |
|---------|---------|---------|
| `JITTER_PREBUFFER_FRAMES` | `6` | Frames buffered before playback (higher = smoother / more latency) |
| `CLIP_LEAD` | `1` | Clips buffered before playback starts (clips mode) |
| `TARGET_FPS` | `25` | Playback rate (matches FlashHead's fixed output) |

---

## Wire protocol (server ↔ browser)

**Browser → server**
- binary = 16kHz mic PCM (int16)
- text JSON `{t: <timestamp>}` = latency ping

**Server → browser**
- `0x01` + JPEG = video frame (frames mode)
- `0x02` + PCM = 24kHz audio (frames mode)
- `0x03` + MP4 = video clip (clips mode)
- text = ping echo

---

## Tuning cheatsheet

**Want lower latency?** Use `frames` mode. Lower `JITTER_PREBUFFER_FRAMES` (e.g. 4)
and `PREBUFFER_FRAMES` (e.g. 6). Pick a pod close to you (watch `net`).

**Want maximum smoothness?** Use `clips` mode. Larger `CLIP_CHUNKS` (3) = smoother
but more startup delay.

**Avatar freezes between turns?** Ensure `ENABLE_IDLE = True`.

**Video lags seconds behind audio (frames mode)?** The queue is growing. Confirm idle
isn't piling on (`IDLE_QUEUE_MAX` low) and that turns are short (the system prompt
enforces 1–3 sentences so the queue drains between turns).

**Generation too slow (`[latency]` fps low)?** That's a GPU/compute issue, not a
network one — a faster GPU (4090) helps; pod distance does not.

---

## Troubleshooting

- **Page won't load / spins forever:** the server port isn't reachable. Confirm the
  pod actually exposes the port you're opening (provider port-mapping), and that the
  server prints `Uvicorn running on 0.0.0.0:7860`. Test on the pod with
  `curl -s http://localhost:7860/ | head`.
- **Can't see/hear anything but the page loads:** check the browser console (F12) and
  the metrics line — if `recv` climbs but `shown` is 0, it's a rendering issue; if
  `recv` is 0, frames aren't arriving.
- **`mediapipe==0.10.9` install error:** apply the requirements fix above.
- **flash-attn "not a supported wheel":** you're on the wrong Python (the wheel is
  cp311). Use a Python 3.11 environment.
- **First response takes a while:** the first generation triggers a one-time
  torch.compile (~4–5 min). Wait for `Server ready.`

---

## License / credits

Built on [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) and the
Google Gemini Live API. Respect the upstream model and API licenses.
