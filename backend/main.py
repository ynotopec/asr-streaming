"""
Nemotron 3.5 ASR Streaming Backend
Real-time cache-aware streaming speech recognition via WebSocket.

Architecture: one `model.generate()` per WebSocket session, fed by a live
generator of `input_features` chunks. This preserves the model's streaming
cache across chunks (the whole point of a cache-aware ASR model) and yields
coherent, incrementally-growing transcripts.
"""

# ── Exclude global Hermes venv from sys.path ─────────────────
import sys
_hermes_site = "/home/ai-agent/.hermes/installs"
sys.path = [p for p in sys.path if not p.startswith(_hermes_site)]

import asyncio
import json
import logging
import queue
import threading
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import (
    Nemotron3_5AsrProcessor,
    Nemotron3_5AsrForRNNT,
    TextIteratorStreamer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
SAMPLE_RATE = 16000

# Only one generation runs at a time (single GPU, single demo user).
GENERATION_LOCK = threading.Lock()


async def load_model():
    global model, processor
    logger.info(f"Loading {MODEL_ID} on {device}…")
    processor = Nemotron3_5AsrProcessor.from_pretrained(MODEL_ID)
    model = Nemotron3_5AsrForRNNT.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else "cpu",
    )
    model.eval()
    logger.info(
        f"Model loaded ✓ | supported latencies (ms): {processor.supported_streaming_latencies_ms}"
    )


def _lookahead_for_latency(latency_ms: int) -> int:
    """Map a requested latency (ms) to the matching num_lookahead_tokens."""
    mapping = processor.supported_streaming_latencies_ms  # {tokens: ms}
    # exact match first
    for tokens, ms in mapping.items():
        if ms == latency_ms:
            return tokens
    # fall back to the closest supported latency
    tokens = min(mapping, key=lambda t: abs(mapping[t] - latency_ms))
    logger.warning(f"Latency {latency_ms}ms unsupported, using {mapping[tokens]}ms")
    return tokens


class ASRStreamSession:
    """A single live streaming transcription session (one model.generate call)."""

    def __init__(self, language: str, latency_ms: int, loop: asyncio.AbstractEventLoop):
        self.language = language
        self.latency_ms = latency_ms
        self.loop = loop

        self.audio_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self.samples = np.zeros(0, dtype=np.float32)   # full session buffer
        self.text = ""                                  # cumulative transcript
        self.text_q: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

        self._gen_thread: Optional[threading.Thread] = None
        self._read_thread: Optional[threading.Thread] = None
        self._closed = False

    # ── public API (called from the event loop) ──────────────
    def push(self, arr: np.ndarray):
        if not self._closed:
            self.audio_q.put(arr)

    def start(self):
        self._gen_thread = threading.Thread(target=self._run, daemon=True)
        self._gen_thread.start()

    def close(self):
        self._closed = True
        self.audio_q.put(None)  # unblock generator → generate() finishes

    # ── worker thread ────────────────────────────────────────
    def _pull_until(self, n: int) -> bool:
        """Block until the session buffer holds >= n samples. False if closed."""
        while self.samples.shape[0] < n:
            item = self.audio_q.get()
            if item is None:
                return False
            self.samples = np.concatenate([self.samples, item])
        return True

    def _run(self):
        try:
            GENERATION_LOCK.acquire()
            self._run_inner()
        except Exception as e:
            import traceback
            logger.error(f"Session error: {e}\n{traceback.format_exc()}")
            self.loop.call_soon_threadsafe(self.text_q.put_nowait, None)
        finally:
            try:
                GENERATION_LOCK.release()
            except RuntimeError:
                pass

    def _run_inner(self):
        processor.set_num_lookahead_tokens(self._lookahead())
        ff = processor.feature_extractor
        hop = ff.hop_length
        nfft = ff.n_fft
        first_need = processor.num_samples_first_audio_chunk
        per_need = processor.num_samples_per_audio_chunk
        first_mel = processor.num_mel_frames_first_audio_chunk
        per_mel = processor.num_mel_frames_per_audio_chunk

        logger.info(
            f"Session start: lang={self.language}, latency={processor.streaming_latency_ms}ms, "
            f"first={first_need} samples, per={per_need} samples"
        )

        # ── wait for the first chunk, build the "first_chunk_inputs" ──
        if not self._pull_until(first_need):
            self.loop.call_soon_threadsafe(self.text_q.put_nowait, None)
            return

        first_inputs = processor(
            self.samples[:first_need],
            sampling_rate=SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=True,
            language=self.language,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        # ── generator yielding input_features chunk after chunk ──
        def input_features_generator():
            yield first_inputs.input_features[:, :first_mel, :]
            mel_idx = first_mel
            start = mel_idx * hop - nfft // 2
            while True:
                end = start + per_need
                if not self._pull_until(end):
                    return
                seg = self.samples[start:end]
                inputs = processor(
                    seg,
                    sampling_rate=SAMPLE_RATE,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                    language=self.language,
                    return_tensors="pt",
                ).to(model.device, dtype=model.dtype)
                yield inputs.input_features
                mel_idx += per_mel
                start = mel_idx * hop - nfft // 2

        streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)

        def consume_streamer():
            last_sent = ""
            try:
                for chunk in streamer:
                    self.text += chunk
                    if self.text != last_sent:      # dedupe: skip identical frames
                        last_sent = self.text
                        self.loop.call_soon_threadsafe(self.text_q.put_nowait, self.text)
            except Exception as e:
                logger.error(f"Streamer error: {e}")
            finally:
                self.loop.call_soon_threadsafe(self.text_q.put_nowait, None)

        self._read_thread = threading.Thread(target=consume_streamer, daemon=True)
        self._read_thread.start()

        generate_kwargs = {
            **dict(first_inputs),
            "input_features": input_features_generator(),
            "streamer": streamer,
        }

        with torch.no_grad():
            model.generate(**generate_kwargs)

        self._read_thread.join(timeout=10)

    def _lookahead(self) -> int:
        mapping = processor.supported_streaming_latencies_ms
        for tokens, ms in mapping.items():
            if ms == self.latency_ms:
                return tokens
        return min(mapping, key=lambda t: abs(mapping[t] - self.latency_ms))


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="Nemotron 3.5 ASR Streaming", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup():
    await load_model()


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": device,
        "dtype": str(dtype),
        "supported_latencies_ms": sorted(processor.supported_streaming_latencies_ms.values())
        if processor
        else [],
    }


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    """
    Protocol:
      1. text JSON config: {"language": "fr-FR", "latency_ms": 560}
      2. binary frames: raw little-endian int16 mono PCM @ 16 kHz
      → text JSON back: {"text": "<cumulative transcript>", "is_final": false|true}
    """
    await ws.accept()
    logger.info("Client connected")

    loop = asyncio.get_running_loop()
    session: Optional[ASRStreamSession] = None
    forward_task: Optional[asyncio.Task] = None

    async def forward():
        assert session is not None
        while True:
            item = await session.text_q.get()
            if item is None:
                await ws.send_json({"text": session.text, "is_final": True})
                break
            await ws.send_json({"text": item, "is_final": False})

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                logger.info("Client disconnected")
                break
            if msg["type"] != "websocket.receive":
                continue

            if "text" in msg and msg["text"] is not None:
                try:
                    cfg = json.loads(msg["text"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if cfg.get("command") == "stop":
                    if session is not None:
                        session.close()  # flush tail + emit is_final
                    continue
                language = cfg.get("language", "auto")
                latency_ms = int(cfg.get("latency_ms", 560))
                if session is None:
                    session = ASRStreamSession(language, latency_ms, loop)
                    forward_task = asyncio.create_task(forward())
                    session.start()
                    logger.info(f"Config: language={language}, latency={latency_ms}ms")
                continue

            if "bytes" in msg and msg["bytes"]:
                if session is None:
                    continue
                arr = np.frombuffer(msg["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                session.push(arr)
                continue

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        import traceback
        logger.error(f"WebSocket error: {e}\n{traceback.format_exc()}")
    finally:
        if session is not None:
            session.close()
        if forward_task is not None:
            try:
                await asyncio.wait_for(forward_task, timeout=5)
            except Exception:
                forward_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8740)
