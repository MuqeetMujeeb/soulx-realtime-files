"""
Real-time SoulX-FlashHead + Gemini Live server (WebSocket transport).
/workspace/SoulX-FlashHead/flashhead_gemini_server.py

Pipeline:
  Browser mic --WebSocket binary--> server (16k PCM int16)
    --> Gemini Live (STS, audio in / audio out)
    --> resample 24k -> 16k
    --> FlashHeadAudioFeeder: rolling 8s buffer, fire pipeline every slice_len
    --> run_pipeline() returns [T,H,W,3] float 0-255 RGB
    --> drop motion frames, JPEG-encode each (RGB->BGR)
    --> WebSocket binary (0x01=video, 0x02=audio) --> browser

Adapted from the working Ditto+Gemini server. Only the audio->frames core
changed (DittoAudioFeeder -> FlashHeadAudioFeeder); Gemini bridge, transport,
and client are unchanged.
"""
import asyncio
import os
import threading
from collections import deque

import numpy as np
from scipy.signal import resample_poly

import cv2
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from flash_head.inference import (
    get_pipeline,
    get_base_data,
    get_audio_embedding,
    run_pipeline,
    get_infer_params,
)

from google import genai
from google.genai import types


# ======================================================================
# Configuration
# ======================================================================
CKPT_DIR     = "models/SoulX-FlashHead-1_3B"
WAV2VEC_DIR  = "models/wav2vec2-base-960h"
MODEL_TYPE   = "lite"
COND_IMAGE   = "examples/prototype.png"   # <-- swap your own 512-ish portrait here
BASE_SEED    = 9999
USE_FACE_CROP = True

# --- Body compositing (head-on-static-body) -----------------------------------
# Segments the person out of each FlashHead frame (MediaPipe Selfie Segmentation)
# and composites it onto a static body image at an adjustable position/scale, with
# a feathered edge to soften the neck seam. Tune the knobs by eye and restart.
ENABLE_COMPOSITE = False                  # master on/off (compare with/without)
BODY_IMAGE       = "examples/prototype.png"   # <-- your static body photo goes here
COMPOSITE_X      = 0                      # horizontal offset of head center (px, +right)
COMPOSITE_Y      = 0                      # vertical offset of head top (px, +down)
COMPOSITE_SCALE  = 1.0                    # scale the FlashHead frame before pasting
FEATHER_PX       = 25                     # soften the mask edge by this many px (seam blend)
OUTPUT_W         = 512                    # final composited frame width  (browser canvas)
OUTPUT_H         = 512                    # final composited frame height

GEMINI_MODEL = "gemini-3.1-flash-live-preview"
GEMINI_SR    = 24000
JPEG_QUALITY = 75

# --- Video delivery mode -------------------------------------------------------
#   "frames" = stream individual JPEG frames; browser jitter-buffers and paints
#              at 25fps. LOW LATENCY (sub-second), needs client smoothing.
#   "clips"  = batch CLIP_CHUNKS chunks into a short MP4 (audio muxed in), send the
#              whole MP4; browser plays it with the native <video> player. VERY
#              SMOOTH but adds ~CLIP_CHUNKS*1.1s latency (must wait for a full clip).
VIDEO_MODE   = "clips"     # "frames" or "clips"  <-- flip to compare
CLIP_CHUNKS  = 3            # clips mode: chunks per MP4 (1≈1.1s ... bigger=smoother/laggier)

# --- Idle motion (keep avatar alive during silence) ---
ENABLE_IDLE     = True     # feed low-noise during gaps so the face doesn't freeze
IDLE_GAP_SEC    = 0.4      # start idle generation after this much silence from Gemini

# --- Output pacing / latency (frames mode) ---
PREBUFFER_FRAMES    = 8     # frames buffered before playback starts
IDLE_QUEUE_MAX      = 6     # only generate idle frames when queue is below this

# Optional persona for Gemini. Set to None to disable.
SYSTEM_INSTRUCTION = (
    "You are a friendly conversational avatar. Keep responses concise, "
    "one to three sentences, and speak naturally."
)


# ======================================================================
# Load FlashHead once at startup
# ======================================================================
print("Loading SoulX-FlashHead pipeline (this takes ~30-60s)...")
pipeline = get_pipeline(
    world_size=1,
    ckpt_dir=CKPT_DIR,
    model_type=MODEL_TYPE,
    wav2vec_dir=WAV2VEC_DIR,
)
print(f"Preparing base data from conditioning image: {COND_IMAGE}")
get_base_data(
    pipeline,
    cond_image_path_or_dir=COND_IMAGE,
    base_seed=BASE_SEED,
    use_face_crop=USE_FACE_CROP,
)

_ip = get_infer_params()
SAMPLE_RATE          = _ip["sample_rate"]            # 16000
TGT_FPS              = _ip["tgt_fps"]                # 25
CACHED_AUDIO_DUR     = _ip["cached_audio_duration"]  # 8 (seconds)
FRAME_NUM            = _ip["frame_num"]              # 33
MOTION_FRAMES_NUM    = _ip["motion_frames_num"]      # 5 (computed in get_pipeline)
SLICE_LEN            = FRAME_NUM - MOTION_FRAMES_NUM  # 28 new frames per chunk

CACHED_AUDIO_SAMPLES = SAMPLE_RATE * CACHED_AUDIO_DUR             # 128000 (8s)
SLICE_SAMPLES        = SLICE_LEN * SAMPLE_RATE // TGT_FPS         # 17920 (~1.12s)
AUDIO_END_IDX        = CACHED_AUDIO_DUR * TGT_FPS                 # 200
AUDIO_START_IDX      = AUDIO_END_IDX - FRAME_NUM                  # 167

print(f"[flashhead] fps={TGT_FPS} frame_num={FRAME_NUM} motion={MOTION_FRAMES_NUM} "
      f"slice_len={SLICE_LEN} slice_samples={SLICE_SAMPLES} "
      f"cached_samples={CACHED_AUDIO_SAMPLES}")
print("FlashHead ready.")


# ======================================================================
# Frame / audio queues + globals
# ======================================================================
frame_queue: asyncio.Queue = None
audio_out_queue: asyncio.Queue = None
clip_queue: asyncio.Queue = None
main_loop: asyncio.AbstractEventLoop = None
audio_feeder = None
compositor = None


# ======================================================================
# Audio bridge
# ======================================================================
def resample_24k_to_16k(audio_int16: np.ndarray) -> np.ndarray:
    f32 = audio_int16.astype(np.float32) / 32768.0
    out = resample_poly(f32, up=2, down=3)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


class BodyCompositor:
    """Segments the person out of each FlashHead frame and composites them onto a
    static body image. Head animates; body is a fixed photo. The neck seam is
    softened by feathering the segmentation mask edge.

    Tune COMPOSITE_X/Y/SCALE/FEATHER_PX by eye. This is the practical 'head on a
    still body' approach: it will not sync body motion to the head (body is static),
    and the seam quality depends on the feather + a body photo whose neck area
    roughly matches where the head sits.
    """
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        # model_selection=1 = general (full-range) selfie segmentation
        self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        # Load + size the static body image once (BGR for cv2 pipeline).
        body = cv2.imread(BODY_IMAGE, cv2.IMREAD_COLOR)
        if body is None:
            raise FileNotFoundError(
                f"BODY_IMAGE not found: {BODY_IMAGE} (set ENABLE_COMPOSITE=False "
                f"to run without compositing, or place your body photo there)")
        self.body = cv2.resize(body, (OUTPUT_W, OUTPUT_H))
        print(f"[composite] body image loaded: {BODY_IMAGE} -> {OUTPUT_W}x{OUTPUT_H}")

    def composite(self, frame_rgb: np.ndarray) -> np.ndarray:
        """frame_rgb: HxWx3 uint8 RGB FlashHead frame. Returns composited RGB."""
        # MediaPipe expects RGB; FlashHead frames are RGB already.
        h, w = frame_rgb.shape[:2]
        res = self.segmenter.process(frame_rgb)
        mask = res.segmentation_mask  # float32 HxW in [0,1], 1=person
        if mask is None:
            # segmentation failed; fall back to the raw frame on the body center
            mask = np.ones((h, w), dtype=np.float32)

        # Feather the mask edge to soften the seam.
        m = (mask * 255).astype(np.uint8)
        if FEATHER_PX > 0:
            k = FEATHER_PX | 1   # odd kernel
            m = cv2.GaussianBlur(m, (k, k), 0)
        alpha = (m.astype(np.float32) / 255.0)[..., None]  # HxWx1

        # Optionally scale the FlashHead frame (and its alpha) before pasting.
        fg = frame_rgb
        if COMPOSITE_SCALE != 1.0:
            nw, nh = int(w * COMPOSITE_SCALE), int(h * COMPOSITE_SCALE)
            fg = cv2.resize(frame_rgb, (nw, nh))
            alpha = cv2.resize(alpha[..., 0], (nw, nh))[..., None]
        fh, fw = fg.shape[:2]

        # Body canvas (convert stored BGR -> RGB for blending in RGB space).
        canvas = cv2.cvtColor(self.body, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Placement: center horizontally (+ COMPOSITE_X), top at COMPOSITE_Y.
        cx = (OUTPUT_W - fw) // 2 + COMPOSITE_X
        cy = COMPOSITE_Y
        x0, y0 = max(0, cx), max(0, cy)
        x1, y1 = min(OUTPUT_W, cx + fw), min(OUTPUT_H, cy + fh)
        # Corresponding region in the foreground (clip if off-canvas).
        fx0, fy0 = x0 - cx, y0 - cy
        fx1, fy1 = fx0 + (x1 - x0), fy0 + (y1 - y0)
        if x1 <= x0 or y1 <= y0:
            return cv2.cvtColor(self.body, cv2.COLOR_BGR2RGB)

        a = alpha[fy0:fy1, fx0:fx1, :]
        fg_region = fg[fy0:fy1, fx0:fx1, :].astype(np.float32)
        bg_region = canvas[y0:y1, x0:x1, :]
        canvas[y0:y1, x0:x1, :] = a * fg_region + (1.0 - a) * bg_region
        return canvas.astype(np.uint8)


class FlashHeadAudioFeeder:
    """
    Rolling-buffer feeder matching the Gradio streaming app's logic.

    Keeps a deque of the last CACHED_AUDIO_SAMPLES (8s) of 16k float audio,
    primed with leading silence. Each time >= SLICE_SAMPLES of new audio has
    accumulated, runs one pipeline pass over the whole rolling window and emits
    SLICE_LEN (28) new frames as JPEG to frame_queue.

    Runs the GPU pipeline under a lock so concurrent pushes serialize (the model
    is fast enough that this stays ahead of real-time).
    """
    def __init__(self, frame_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop,
                 clip_queue: asyncio.Queue = None):
        self.frame_queue = frame_queue
        self.clip_queue = clip_queue            # used in clips mode
        self.loop = loop
        self.lock = threading.Lock()
        # Rolling buffer primed with silence
        self.audio_dq = deque(
            [0.0] * CACHED_AUDIO_SAMPLES, maxlen=CACHED_AUDIO_SAMPLES
        )
        self.pending = 0  # new samples since last generation
        self.chunks_done = 0
        self.idle_chunks_done = 0
        # event-loop time of the last REAL (Gemini) audio push; idle feeder uses it
        self.last_real_audio_t = 0.0
        # clips mode: accumulate frames + the audio that drove them
        self._clip_frames = []   # list of np frame arrays
        self._clip_audio = []    # list of np audio slices (16k float)
        self._clip_count = 0     # chunks accumulated toward current clip

    def push(self, audio_16k_float32: np.ndarray, is_real: bool = True):
        with self.lock:
            if is_real:
                self.last_real_audio_t = self.loop.time()
            self.audio_dq.extend(audio_16k_float32.tolist())
            self.pending += len(audio_16k_float32)
            fired = 0
            while self.pending >= SLICE_SAMPLES:
                self.pending -= SLICE_SAMPLES
                # the SLICE_SAMPLES of audio that drives this chunk = the newest
                # SLICE_SAMPLES in the rolling buffer
                chunk_audio = np.array(list(self.audio_dq)[-SLICE_SAMPLES:], dtype=np.float32)
                self._generate_one_chunk(chunk_audio, idle=not is_real)
                fired += 1
            if fired and is_real:
                print(f"[feeder] +{len(audio_16k_float32)} samples, "
                      f"fired {fired} chunk(s), total={self.chunks_done}, "
                      f"q={self.frame_queue.qsize()}")

    def seconds_since_real_audio(self) -> float:
        return self.loop.time() - self.last_real_audio_t

    def _generate_one_chunk(self, chunk_audio: np.ndarray, idle: bool = False):
        t0 = self.loop.time()
        audio_array = np.array(self.audio_dq, dtype=np.float32)
        with torch.no_grad():
            emb = get_audio_embedding(
                pipeline, audio_array, AUDIO_START_IDX, AUDIO_END_IDX
            )
            video = run_pipeline(pipeline, emb)        # [T,H,W,3] float 0-255 RGB
            video = video[MOTION_FRAMES_NUM:]          # drop motion-context frames
            frames = video.to(torch.uint8).cpu().numpy()
        if idle:
            self.idle_chunks_done += 1
        else:
            self.chunks_done += 1

        # Composite head-on-body if enabled (each frame: segment + paste on body).
        if ENABLE_COMPOSITE and compositor is not None:
            composited = []
            for f in frames:
                try:
                    composited.append(compositor.composite(f))
                except Exception as e:
                    print(f"[composite] error: {e}")
                    composited.append(f)
            frames = np.stack(composited, axis=0)

        if VIDEO_MODE == "clips":
            # Accumulate frames + driving audio; emit an MP4 every CLIP_CHUNKS.
            self._clip_frames.append(frames)
            self._clip_audio.append(chunk_audio)
            self._clip_count += 1
            if self._clip_count >= CLIP_CHUNKS:
                self._emit_clip()
        else:
            # frames mode: JPEG each frame to the frame queue
            for f in frames:
                bgr = f[..., ::-1]                      # RGB -> BGR for cv2
                ok, buf = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if not ok:
                    continue
                self.loop.call_soon_threadsafe(self._try_put, buf.tobytes())

        if not idle:
            gen_ms = (self.loop.time() - t0) * 1000.0
            print(f"[latency] chunk gen: {gen_ms:.0f}ms for {len(frames)} frames "
                  f"({len(frames)/max(gen_ms/1000,1e-6):.0f} fps)")

    def _emit_clip(self):
        """Encode the accumulated frames + audio into a single MP4 (browser-friendly)
        and push its bytes to the clip queue."""
        import tempfile, subprocess, os as _os
        frames_all = np.concatenate(self._clip_frames, axis=0)   # [N,H,W,3] RGB
        audio_all = np.concatenate(self._clip_audio).astype(np.float32)
        self._clip_frames, self._clip_audio, self._clip_count = [], [], 0

        tmpdir = tempfile.mkdtemp(prefix="clip_")
        raw_mp4 = _os.path.join(tmpdir, "raw.mp4")
        wav_path = _os.path.join(tmpdir, "a.wav")
        out_mp4 = _os.path.join(tmpdir, "out.mp4")
        try:
            # write frames (RGB) to mp4 via imageio
            import imageio
            with imageio.get_writer(raw_mp4, format="mp4", mode="I",
                                    fps=TGT_FPS, codec="h264",
                                    ffmpeg_params=["-bf", "0"]) as w:
                for i in range(frames_all.shape[0]):
                    w.append_data(frames_all[i])
            # write audio wav (16k int16)
            import wave
            samples = (np.clip(audio_all, -1, 1) * 32767).astype(np.int16)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                wf.writeframes(samples.tobytes())
            # mux, browser-friendly flags
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_mp4, "-i", wav_path,
                 "-c:v", "copy", "-c:a", "aac",
                 "-movflags", "+faststart", "-pix_fmt", "yuv420p", out_mp4],
                check=True, capture_output=True)
            with open(out_mp4, "rb") as f:
                clip_bytes = f.read()
            self.loop.call_soon_threadsafe(self._try_put_clip, clip_bytes)
            print(f"[clip] emitted MP4 {len(clip_bytes)//1024}KB "
                  f"({frames_all.shape[0]} frames)")
        except Exception as e:
            print(f"[clip] encode error: {e}")
        finally:
            for p in (raw_mp4, wav_path, out_mp4):
                try: _os.remove(p)
                except Exception: pass
            try: _os.rmdir(tmpdir)
            except Exception: pass

    def _try_put_clip(self, data: bytes):
        try:
            self.clip_queue.put_nowait(data)
        except asyncio.QueueFull:
            pass

    def push_idle(self):
        """Feed one slice of low-amplitude noise to keep the avatar alive during
        silence. The model is trained on silence-padded buffers, so this yields a
        subtly-moving/settling face rather than a hard freeze. Low amplitude so it
        reads as 'quiet/listening', not talking."""
        # Tiny noise floor (~ -60 dBFS) so the face has something to animate from
        # without producing mouth movement that looks like speech.
        noise = (np.random.randn(SLICE_SAMPLES).astype(np.float32)) * 0.001
        self.push(noise, is_real=False)

    def _try_put(self, data: bytes):
        try:
            self.frame_queue.put_nowait(data)
        except asyncio.QueueFull:
            pass  # drop frame if browser can't keep up

    def push_silence_ms(self, ms: int):
        n = int(ms * SAMPLE_RATE / 1000)
        self.push(np.zeros(n, dtype=np.float32))


# ======================================================================
# Gemini Live
# ======================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY env var")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def _make_live_config():
    kwargs = {"response_modalities": ["AUDIO"]}
    if SYSTEM_INSTRUCTION:
        kwargs["system_instruction"] = types.Content(
            parts=[types.Part(text=SYSTEM_INSTRUCTION)]
        )
    return types.LiveConnectConfig(**kwargs)


async def gemini_session_with_counter(user_audio_queue, stop_event, counters):
    config = _make_live_config()
    print("[gemini] connecting...")
    try:
        async with gemini_client.aio.live.connect(
            model=GEMINI_MODEL, config=config
        ) as session:
            print("[gemini] connected to Live API")

            # Mic pump runs for the ENTIRE session lifetime, across all turns.
            async def pump_user_audio():
                while not stop_event.is_set():
                    try:
                        pcm16k = await asyncio.wait_for(
                            user_audio_queue.get(), timeout=0.5
                        )
                    except asyncio.TimeoutError:
                        continue
                    try:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=pcm16k.tobytes(),
                                mime_type="audio/pcm;rate=16000",
                            )
                        )
                    except Exception as e:
                        # Don't kill the pump on a transient send error.
                        print(f"[gemini] send error (continuing): {e}")
                        await asyncio.sleep(0.05)
                        continue

            pump_task = asyncio.create_task(pump_user_audio())
            try:
                # OUTER loop: each session.receive() generator covers ONE turn.
                # When it completes (turn_complete), loop back to receive the
                # next turn. This keeps the conversation going indefinitely.
                while not stop_event.is_set():
                    async for response in session.receive():
                        if stop_event.is_set():
                            break
                        if getattr(response, "data", None):
                            if len(response.data) < 320:
                                continue  # skip suspicious tiny blocks
                            counters["gemini_audio_chunks"] += 1
                            if counters["gemini_audio_chunks"] == 1:
                                print(f"[gemini] FIRST real audio, {len(response.data)} bytes")
                            gemini_pcm = np.frombuffer(response.data, dtype=np.int16)
                            fh_audio = resample_24k_to_16k(gemini_pcm)
                            # Feed model (blocking GPU work -> thread) ...
                            await asyncio.to_thread(audio_feeder.push, fh_audio)
                            # ... and (frames mode only) queue 24k audio for browser
                            # playback. In clips mode audio is muxed into the MP4.
                            if VIDEO_MODE == "frames":
                                try:
                                    audio_out_queue.put_nowait(response.data)
                                except asyncio.QueueFull:
                                    pass
                        elif getattr(response, "text", None):
                            print(f"[gemini] text: {response.text!r}")
                        sc = getattr(response, "server_content", None)
                        if sc and getattr(sc, "turn_complete", False):
                            print("[gemini] turn_complete")

                    # The async-for ended -> this turn is over. Loop back and
                    # call session.receive() again for the next turn. The pump
                    # keeps streaming mic audio, so Gemini's VAD picks up the
                    # user's next utterance and starts a new turn.
                    print("[gemini] turn ended; listening for next turn...")
            except Exception as e:
                print(f"[gemini] receive error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                pump_task.cancel()
    except Exception as e:
        print(f"[gemini] CONNECT FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# ======================================================================
# FastAPI + WebSocket
# ======================================================================
app = FastAPI()


@app.on_event("startup")
async def on_startup():
    global frame_queue, audio_out_queue, clip_queue, main_loop, audio_feeder, compositor
    main_loop = asyncio.get_running_loop()
    frame_queue = asyncio.Queue(maxsize=120)   # backpressure caps real depth ~30
    audio_out_queue = asyncio.Queue(maxsize=300)
    clip_queue = asyncio.Queue(maxsize=30)     # clips mode: finished MP4s
    audio_feeder = FlashHeadAudioFeeder(frame_queue, main_loop, clip_queue)
    if ENABLE_COMPOSITE:
        print("Initializing body compositor (MediaPipe selfie segmentation)...")
        compositor = BodyCompositor()
    print(f"VIDEO_MODE = {VIDEO_MODE}" + (f" (CLIP_CHUNKS={CLIP_CHUNKS})" if VIDEO_MODE == "clips" else "")
          + f"  COMPOSITE = {ENABLE_COMPOSITE}")
    print("Priming pipeline with silence (also triggers torch.compile warmup)...")
    # One slice of silence to JIT-compile the model before the first user turn.
    audio_feeder.push_silence_ms(int(SLICE_SAMPLES * 1000 / SAMPLE_RATE) + 50)
    print("Server ready.")


@app.get("/")
async def index():
    return HTMLResponse(open("client.html").read())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("=" * 50)
    print("WebSocket connected")
    print("=" * 50)
    stop_event = asyncio.Event()
    user_audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    counters = {"mic_chunks": 0, "gemini_audio_chunks": 0, "frames_sent": 0}

    async def stats_logger():
        while not stop_event.is_set():
            await asyncio.sleep(2.0)
            print(f"[stats] mic={counters['mic_chunks']} "
                  f"gemini_audio={counters['gemini_audio_chunks']} "
                  f"frames_sent={counters['frames_sent']} "
                  f"frame_q={frame_queue.qsize()} "
                  f"user_audio_q={user_audio_queue.qsize()} "
                  f"chunks={audio_feeder.chunks_done} "
                  f"idle={audio_feeder.idle_chunks_done}")

    gemini_task = asyncio.create_task(
        gemini_session_with_counter(user_audio_queue, stop_event, counters)
    )
    stats_task = asyncio.create_task(stats_logger())

    async def receive_audio():
        try:
            while not stop_event.is_set():
                msg = await ws.receive()
                # A ping arrives as TEXT (JSON with the browser's timestamp); echo
                # it straight back so the browser can compute round-trip latency
                # against its own clock (no server/browser clock-sync needed).
                if msg.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(0)
                if msg.get("text") is not None:
                    # echo the ping back verbatim, tagged 0x03 in a text frame
                    try:
                        await ws.send_text(msg["text"])
                    except Exception:
                        pass
                    continue
                data = msg.get("bytes")
                if data is None:
                    continue
                pcm = np.frombuffer(data, dtype=np.int16)
                counters["mic_chunks"] += 1
                if counters["mic_chunks"] == 1:
                    print(f"[server] FIRST mic chunk, samples={len(pcm)}")
                try:
                    user_audio_queue.put_nowait(pcm)
                except asyncio.QueueFull:
                    pass
        except WebSocketDisconnect:
            print("WS disconnected (audio)")
            stop_event.set()

    async def idle_loop():
        """Keep the avatar alive during silence. When Gemini hasn't sent audio
        for IDLE_GAP_SEC, feed low-noise slices so the model keeps generating a
        subtly-moving face instead of freezing on the last frame. Paced ~real-time
        so it doesn't hog the GPU from speech turns."""
        if not ENABLE_IDLE:
            return
        # one idle slice ~= SLICE_LEN frames of video; pace so we generate roughly
        # in step with playback (don't outrun it and pile up idle frames)
        idle_interval = SLICE_LEN / TGT_FPS  # ~1.12s of video per idle slice
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
            if audio_feeder.seconds_since_real_audio() < IDLE_GAP_SEC:
                continue
            # Only generate idle when the relevant output queue is nearly empty,
            # i.e. real speech video has drained and we'd otherwise freeze.
            q_depth = (clip_queue.qsize() if VIDEO_MODE == "clips"
                       else frame_queue.qsize())
            idle_thresh = 1 if VIDEO_MODE == "clips" else IDLE_QUEUE_MAX
            if q_depth < idle_thresh:
                await asyncio.to_thread(audio_feeder.push_idle)
                await asyncio.sleep(idle_interval * 0.5)

    async def send_outputs():
        """frames mode: steady 25fps frame pacing (below).
        clips mode: just forward finished MP4s (0x03) as they're produced."""
        if VIDEO_MODE == "clips":
            try:
                while not stop_event.is_set():
                    try:
                        clip = clip_queue.get_nowait()
                        await ws.send_bytes(b"\x03" + clip)
                        counters["frames_sent"] += 1
                        if counters["frames_sent"] == 1:
                            print(f"[server] FIRST clip sent, size={len(clip)//1024}KB")
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.02)
            except WebSocketDisconnect:
                print("WS disconnected (output)")
                stop_event.set()
            except Exception as e:
                print(f"Send error: {e}")
                stop_event.set()
            return

        # ---- frames mode (default) ----
        FRAME_INTERVAL = 1.0 / TGT_FPS          # 40ms/frame, strict
        started = False
        next_frame_time = None
        try:
            while not stop_event.is_set():
                # Audio out immediately.
                try:
                    audio_bytes = audio_out_queue.get_nowait()
                    await ws.send_bytes(b"\x02" + audio_bytes)
                except asyncio.QueueEmpty:
                    pass

                if not started:
                    if frame_queue.qsize() >= PREBUFFER_FRAMES:
                        started = True
                        next_frame_time = asyncio.get_event_loop().time()
                    else:
                        await asyncio.sleep(0.005)
                        continue

                try:
                    jpeg_bytes = frame_queue.get_nowait()
                    counters["frames_sent"] += 1
                    if counters["frames_sent"] == 1:
                        print(f"[server] FIRST frame sent, size={len(jpeg_bytes)}")
                    await ws.send_bytes(b"\x01" + jpeg_bytes)
                    # Strict 25fps cadence - smooth, never bursts.
                    next_frame_time += FRAME_INTERVAL
                    sleep_for = next_frame_time - asyncio.get_event_loop().time()
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                    else:
                        # We're behind the clock (e.g. a slow send); don't sleep,
                        # but DON'T burst either - just realign the clock so we
                        # resume steady 40ms pacing from now.
                        next_frame_time = asyncio.get_event_loop().time()
                except asyncio.QueueEmpty:
                    # Nothing to send right now; if fully empty, re-arm prebuffer.
                    if frame_queue.qsize() == 0:
                        started = False
                    await asyncio.sleep(0.005)
        except WebSocketDisconnect:
            print("WS disconnected (output)")
            stop_event.set()
        except Exception as e:
            print(f"Send error: {e}")
            stop_event.set()

    idle_task = asyncio.create_task(idle_loop())

    try:
        await asyncio.gather(receive_audio(), send_outputs())
    finally:
        stop_event.set()
        gemini_task.cancel()
        stats_task.cancel()
        idle_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")