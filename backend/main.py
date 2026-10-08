"""
Audio-to-Audio Translator — Full Pipeline
ASR (Nemotron 3.5 streaming) → Text Translation (MarianMT) → TTS (Qwen3-TTS)
"""

import sys
sys.path = [p for p in sys.path if not p.startswith("/home/ai-agent/.hermes/installs")]
sys.path.insert(0, "/home/ai-agent/work/audio2audio-translate-next/backend")

import asyncio
import json
import logging
import threading
import tempfile
import os
import io
from typing import Optional

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

import tts_engine as _tts_engine
from translation import translate_text, init_translation, SUPPORTED_LANGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Globals ─────────────────────────────────────────
nemo_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

ASR_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
SAMPLE_RATE = 16000

# ── FastAPI app ─────────────────────────────────────
app = FastAPI(title="Audio2Audio Translator", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


def trim_silence(audio: np.ndarray, sr: int, threshold: float = 1e-3, pad_ms: int = 60) -> np.ndarray:
    """Trim leading/trailing silence from TTS output."""
    if len(audio) == 0:
        return audio
    win = max(1, sr // 100)
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(win) / win, mode="same"))
    voiced = np.where(rms > threshold)[0]
    if len(voiced) == 0:
        return audio
    pad = sr * pad_ms // 1000
    return audio[max(0, voiced[0] - pad): min(len(audio), voiced[-1] + pad)]


def audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


import re
_LANG_TAG_RE = re.compile(r"\s*<([a-z]{2}(?:-[A-Za-z]{2})?)>\s*$")

def parse_transcript(raw: str) -> tuple[str, str | None]:
    """Strip Nemotron language tag `<en-US>`; return (text, detected_locale|None)."""
    m = _LANG_TAG_RE.search(raw)
    text = _LANG_TAG_RE.sub("", raw).strip()
    return (text, m.group(1) if m else None)


def transcribe_nemotron(audio: np.ndarray) -> tuple[str, str | None]:
    """Nemotron ASR in-memory (numpy) — no temp file, no dataloader overhead."""
    if nemo_model is None:
        return "", None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if len(audio) < SAMPLE_RATE // 4:  # <250ms: nothing usable
        return "", None
    try:
        result = nemo_model.transcribe([audio], verbose=False)
        if not result:
            return "", None
        first = result[0]
        if isinstance(first, str):
            raw = first
        elif isinstance(first, dict):
            raw = first.get("text", "") or first.get("pred_text", "")
        else:
            raw = getattr(first, "text", "") or ""
        return parse_transcript(raw)
    except Exception as e:
        logger.error(f"ASR error: {e}")
        return "", None


# ── Batch API ────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "asr_model": ASR_MODEL_ID if nemo_model else ASR_MODEL_ID + " (not loaded)",
        "tts_model": _tts_engine.model.__class__.__name__ if _tts_engine.model else "not loaded",
        "device": device,
        "supported_languages": list(SUPPORTED_LANGS.keys()),
    }


@app.post("/api/translate")
async def translate_file(
    file: UploadFile = File(...),
    src_lang: str = Form("auto"),
    tgt_lang: str = Form("fr-FR"),
):
    """Batch: upload audio → translated WAV."""
    content = await file.read()

    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(file.filename)[1] or ".wav", delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        audio_np, sr = sf.read(tmp_path)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != SAMPLE_RATE:
            ratio = SAMPLE_RATE / sr
            new_len = int(len(audio_np) * ratio)
            audio_np = np.interp(
                np.linspace(0, len(audio_np) - 1, new_len),
                np.arange(len(audio_np)),
                audio_np,
            )

        # ASR with Nemotron (auto-detected language tag available)
        logger.info(f"ASR: {src_lang} → translate → {tgt_lang}")
        transcript, detected = transcribe_nemotron(audio_np)
        effective_src = src_lang if src_lang != "auto" else (detected or "en-US")
        logger.info(f"Transcript: {transcript} (detected: {detected})")

        # Translate
        translated = translate_text(transcript, effective_src, tgt_lang)
        logger.info(f"Translated: {translated}")

        # TTS
        if _tts_engine.model is not None:
            wavs, tts_sr = await _tts_engine.synthesize(translated, tgt_lang)
            wav_bytes = audio_to_wav_bytes(trim_silence(np.asarray(wavs, dtype=np.float32), tts_sr), tts_sr)
        else:
            wav_bytes = b"Error: TTS model not loaded"

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as e:
        import traceback
        logger.error(f"Batch translate error: {e}\n{traceback.format_exc()}")
        return Response(content=f"Error: {str(e)}", status_code=500)
    finally:
        os.unlink(tmp_path)


# ── Streaming WebSocket ──────────────────────────────
@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    """
    Streaming audio→text→audio.
      1. JSON config: {src_lang, tgt_lang, latency_ms}
      2. Binary PCM from mic
      ← JSON {"type":"transcript","text":"..."}
      ← JSON {"type":"translated","text":"..."}
      ← Binary WAV (TTS)
    """
    await ws.accept()
    logger.info("Client connected to translation WS")

    config = {"src_lang": "auto", "tgt_lang": "fr-FR", "sample_rate": 16000}
    audio_chunks: list[np.ndarray] = []
    buffer_lock = threading.Lock()
    stop_event = threading.Event()

    async def _flush_and_respond(text: str, detected: str | None):
        effective_src = config["src_lang"] if config["src_lang"] != "auto" else (detected or "en-US")
        translated = translate_text(text, effective_src, config["tgt_lang"])
        await ws.send_json({"type": "transcript", "text": text})
        await ws.send_json({"type": "translated", "text": translated})
        # Skip TTS for fragments (<4 words): no useful speech, avoids silence gaps
        if len(text.split()) < 4:
            logger.info(f"TTS skipped (fragment, {len(text.split())} words): {text!r}")
            return
        if _tts_engine.model is not None:
            wavs, sr = await _tts_engine.synthesize(translated, config["tgt_lang"])
            wav_bytes = audio_to_wav_bytes(trim_silence(np.asarray(wavs, dtype=np.float32), sr), sr)
            await ws.send_bytes(wav_bytes)

    def _asr_thread():
        """
        Pause-based segmentation: accumulate mic audio, cut at natural
        pauses (>=300ms below energy floor) or hard limit 3s. Segments
        align on word boundaries — no mid-syllable cuts, no silence ASR.
        """
        frame = SAMPLE_RATE // 10          # 100ms analysis frame
        pause_frames = 3                   # 300ms of low energy = pause
        max_seg = SAMPLE_RATE * 3
        min_seg = int(SAMPLE_RATE * 0.8)
        energy_floor = 3e-4                # RMS below this = silence

        buf = np.zeros(0, dtype=np.float32)
        low_energy_run = 0
        try:
            while not stop_event.is_set():
                with buffer_lock:
                    if audio_chunks:
                        buf = np.concatenate([buf] + audio_chunks)
                        audio_chunks.clear()

                do_flush = False
                if len(buf) >= frame:
                    # analyze last frame energy
                    last = buf[-frame:]
                    rms = float(np.sqrt((last ** 2).mean()))
                    if rms < energy_floor:
                        low_energy_run += 1
                        if low_energy_run >= pause_frames and len(buf) >= min_seg:
                            do_flush = True   # flush everything incl. trailing pause
                    else:
                        low_energy_run = 0
                    if len(buf) >= max_seg:
                        do_flush = True

                if do_flush and len(buf) > 0:
                    # trim leading/trailing silence off the segment
                    frame_rms = np.sqrt(np.convolve(buf ** 2, np.ones(frame)/frame, mode="same"))
                    voiced = np.where(frame_rms > energy_floor)[0]
                    if len(voiced) == 0:
                        buf = np.zeros(0, dtype=np.float32)
                        low_energy_run = 0
                        continue
                    seg = buf[max(0, voiced[0] - frame): min(len(buf), voiced[-1] + 2 * frame)]
                    buf = np.zeros(0, dtype=np.float32)
                    low_energy_run = 0
                    text, detected = transcribe_nemotron(seg)
                    if text:
                        asyncio.run_coroutine_threadsafe(
                            _flush_and_respond(text, detected), loop
                        )
                stop_event.wait(0.1)

            # Final flush
            if len(buf) > 0:
                text, detected = transcribe_nemotron(buf)
                if text:
                    asyncio.run_coroutine_threadsafe(_flush_and_respond(text, detected), loop)
        except Exception as e:
            logger.error(f"ASR thread error: {e}")

    loop = asyncio.get_running_loop()
    asr_thread = threading.Thread(target=_asr_thread, daemon=True)
    asr_thread.start()

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
                    stop_event.set()
                    break

                if "src_lang" in cfg:
                    config["src_lang"] = cfg["src_lang"]
                if "tgt_lang" in cfg:
                    config["tgt_lang"] = cfg["tgt_lang"]
                if "sample_rate" in cfg:
                    try:
                        config["sample_rate"] = float(cfg["sample_rate"])
                    except (TypeError, ValueError):
                        pass
                    logger.info(f"Client configured: sr={config['sample_rate']} src={config['src_lang']} tgt={config['tgt_lang']}")
                continue

            if "bytes" in msg and msg["bytes"]:
                with buffer_lock:
                    arr = np.frombuffer(msg["bytes"], dtype=np.int16).astype(np.float32) / 32767.0
                    src_sr = config.get("sample_rate", 16000)
                    if src_sr and abs(src_sr - SAMPLE_RATE) > 1:
                        # resample client rate → pipeline rate (16k)
                        ratio = SAMPLE_RATE / src_sr
                        new_len = max(1, int(len(arr) * ratio))
                        arr = np.interp(
                            np.linspace(0, len(arr) - 1, new_len),
                            np.arange(len(arr)),
                            arr,
                        ).astype(np.float32)
                    audio_chunks.append(arr)
                continue

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_event.set()


# ── Startup ────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global nemo_model

    logger.info(f"Loading {ASR_MODEL_ID} on {device}…")
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    nemo_model = EncDecRNNTBPEModel.from_pretrained(
        model_name=ASR_MODEL_ID,
        map_location=device,
    )
    nemo_model.eval()
    if device == "cuda":
        nemo_model = nemo_model.to(device)
    logger.info(f"Nemotron ASR loaded ✓ on {device}")

    # Prompt-mode fix: fall back to language-agnostic 'auto' prompt when the
    # audio cut carries no language supervision (batch files, mic chunks).
    import nemo.collections.asr.data.audio_to_text_lhotse_prompt_index as _pidx
    _orig_get_prompt = _pidx.LhotseSpeechToTextBpeDatasetWithPromptIndex._get_prompt_index_for_cut
    def _get_prompt_fallback(self, cut):
        mode = self._get_prompt_mode(cut)
        lang = getattr(cut.supervisions[0], "language", None)
        if mode == "langID" and lang:
            try:
                return self._get_prompt_index(lang)
            except ValueError:
                pass
        return self.auto_index
    _pidx.LhotseSpeechToTextBpeDatasetWithPromptIndex._get_prompt_index_for_cut = _get_prompt_fallback
    logger.info("Nemotron prompt-mode fallback (auto) installed ✓")

    init_translation()
    await _tts_engine.load_model()