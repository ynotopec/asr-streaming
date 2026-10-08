"""
Qwen3-TTS audio synthesis backend.
"""
import asyncio
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

LANG_MAP = {
    "en-US": "English", "en-GB": "English",
    "fr-FR": "French", "fr-CA": "French",
    "de-DE": "German",
    "es-ES": "Spanish", "es-US": "Spanish",
    "it-IT": "Italian",
    "pt-BR": "Portuguese", "pt-PT": "Portuguese",
    "zh-CN": "Chinese",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "ru-RU": "Russian",
}

DEFAULT_SPEAKER = {
    "Chinese": "vivian",
    "English": "aiden",
    "Japanese": "ono_anna",
    "Korean": "sohee",
    "German": "dylan",
    "French": "serena",
    "Russian": "eric",
    "Portuguese": "eric",
    "Spanish": "dylan",
    "Italian": "sunny",
}


def _tts_lang(locale_id: str) -> str:
    return LANG_MAP.get(locale_id, "English")


async def load_model():
    global model
    logger.info(f"Loading {MODEL_ID} on {device}…")
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0" if device == "cuda" else None,
        dtype=dtype,
    )
    logger.info("Qwen3-TTS model loaded ✓")


async def synthesize(text: str, language: str, speaker: Optional[str] = None):
    if not text or not text.strip():
        return np.array([], dtype=np.float32), 24000

    tts_lang = _tts_lang(language)
    if speaker is None:
        speaker = DEFAULT_SPEAKER.get(tts_lang, "aiden")

    logger.info(f"TTS: lang={tts_lang}, speaker={speaker}, text_len={len(text)}")

    loop = asyncio.get_running_loop()

    def _run():
        wavs, sr = model.generate_custom_voice(
            text=text.strip(),
            language=tts_lang,
            speaker=speaker,
            instruct="",
        )
        if isinstance(wavs, list):
            wavs = wavs[0]
        if isinstance(wavs, torch.Tensor):
            arr = wavs.detach().cpu().float().numpy()
        else:
            arr = np.array(wavs, dtype=np.float32)
        return arr, sr

    return await loop.run_in_executor(None, _run)


def get_supported_languages() -> list:
    return model.get_supported_languages() if model else []


def get_supported_speakers() -> list:
    return model.get_supported_speakers() if model else []
