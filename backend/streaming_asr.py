"""
Cache-aware streaming ASR for nvidia/nemotron-3.5-asr-streaming-0.6b.

Why this exists: the model is a cache-aware FastConformer. Its encoder keeps
left context in a cache across chunks. Transcribing isolated segments with
`model.transcribe()` throws that context away, which degrades every segment.

This class keeps one encoder cache per utterance and feeds it contiguous
feature chunks (same chunk / pre-encode / shift layout as NeMo's
CacheAwareStreamingAudioBuffer). Utterances are closed on a real pause, not on
a fixed timer, so a word is never cut in half.

Usage:
    s = UtteranceStreamer(model)
    for kind, text in s.push(samples_16k_float32):   # kind: "partial" | "final"
        ...
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import torch

logger = logging.getLogger("streaming_asr")

SR = 16000
FRAME = SR // 10  # 100 ms energy frame
HOP = 160         # feature hop (10 ms) of the Nemotron preprocessor


def _pick(value, first: bool):
    """streaming_cfg values may be [first_step, later_steps] lists."""
    if isinstance(value, (list, tuple)):
        return value[0] if first else value[1]
    return value


class UtteranceStreamer:
    def __init__(
        self,
        model,
        prompt: str = "auto",
        endpoint_silence_s: float = 0.6,
        max_utterance_s: float = 10.0,
        min_noise_rms: float = 1e-3,
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = torch.float32
        self.scfg = model.encoder.streaming_cfg
        if self.scfg is None:
            model.encoder.setup_streaming_params()
            self.scfg = model.encoder.streaming_cfg
        self.n_feat = model.encoder._feat_in

        self.endpoint_frames = int(endpoint_silence_s * 10)  # 100ms frames
        self.max_samples = int(max_utterance_s * SR)
        self.min_noise_rms = min_noise_rms
        self.noise_rms = min_noise_rms

        try:
            model.set_inference_prompt(prompt)
        except Exception as e:  # prompt not in dictionary → model default
            logger.warning(f"prompt '{prompt}' not applied: {e}")

        self._pre_roll = deque(maxlen=3)  # 300ms kept before speech onset
        self._frame_buf = np.zeros(0, dtype=np.float32)
        self._reset_utterance()

    # ── state ────────────────────────────────────────────
    def _reset_utterance(self):
        self.in_speech = False
        self.silent_frames = 0
        self.audio = np.zeros(0, dtype=np.float32)
        self.feats = None
        self.idx = 0
        self.step = 0
        self.prev_hyp = None
        self.last_text = ""
        self.cache_ch, self.cache_t, self.cache_len = self.model.encoder.get_initial_cache_state(
            batch_size=1, dtype=self.dtype, device=self.device
        )

    # ── public API ───────────────────────────────────────
    @torch.inference_mode()
    def push(self, samples: np.ndarray):
        """Feed raw 16 kHz float32 samples. Returns a list of (kind, text)."""
        events = []
        self._frame_buf = np.concatenate([self._frame_buf, samples.astype(np.float32)])
        while len(self._frame_buf) >= FRAME:
            frame, self._frame_buf = self._frame_buf[:FRAME], self._frame_buf[FRAME:]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            threshold = max(self.min_noise_rms, self.noise_rms * 3.0)
            voiced = rms > threshold

            if not self.in_speech:
                self.noise_rms = 0.95 * self.noise_rms + 0.05 * min(rms, threshold)
                self._pre_roll.append(frame)
                if voiced:
                    self.in_speech = True
                    self.silent_frames = 0
                    for f in self._pre_roll:
                        self.audio = np.concatenate([self.audio, f])
                    self._pre_roll.clear()
                continue

            self.audio = np.concatenate([self.audio, frame])
            self.silent_frames = 0 if voiced else self.silent_frames + 1
            ended = self.silent_frames >= self.endpoint_frames or len(self.audio) >= self.max_samples
            if ended:
                events.extend(self._finalize())
                continue

            # enough new audio: decode the new chunks (partial hypothesis)
            text = self._advance(final=False)
            if text is not None and text != self.last_text:
                self.last_text = text
                events.append(("partial", text))
        return events

    # ── internals ────────────────────────────────────────
    def _features(self, wav: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(wav).to(self.device, self.dtype).unsqueeze(0)
        length = torch.tensor([x.shape[1]], device=self.device)
        feats, _ = self.model.preprocessor(input_signal=x, length=length)
        return feats  # (1, D, T)

    def _finalize(self):
        # trailing silence lets the encoder look ahead and the decoder close the word
        tail = np.zeros(int(0.8 * SR), dtype=np.float32)
        self.audio = np.concatenate([self.audio, tail])
        text = self._advance(final=True)
        text = text if text is not None else self.last_text
        out = [("final", text)] if text.strip() else []
        self._reset_utterance()
        return out

    def _advance(self, final: bool):
        self.feats = self._features(self.audio)
        T = self.feats.shape[-1]
        pad = self.model.encoder.streaming_cfg
        last_text = None

        while True:
            first = self.step == 0
            chunk = int(_pick(self.scfg.chunk_size, first))
            shift = int(_pick(self.scfg.shift_size, first))
            pre_n = int(_pick(self.scfg.pre_encode_cache_size, first))
            if self.idx + chunk > T and not final:
                break
            if self.idx >= T:
                break

            audio_chunk = self.feats[:, :, self.idx: self.idx + chunk]
            if first:
                pre = torch.zeros((1, self.n_feat, pre_n), device=self.device, dtype=self.dtype)
            else:
                start = max(0, self.idx - pre_n)
                pre = self.feats[:, :, start: self.idx]
                if pre.shape[-1] < pre_n:
                    pre = torch.cat(
                        [torch.zeros((1, self.n_feat, pre_n - pre.shape[-1]), device=self.device, dtype=self.dtype), pre],
                        dim=-1,
                    )
            x = torch.cat([pre, audio_chunk], dim=-1)
            length = torch.tensor([x.shape[-1]], device=self.device)
            is_last = final and (self.idx + shift >= T)

            out = self.model.conformer_stream_step(
                processed_signal=x,
                processed_signal_length=length,
                cache_last_channel=self.cache_ch,
                cache_last_time=self.cache_t,
                cache_last_channel_len=self.cache_len,
                keep_all_outputs=is_last,
                previous_hypotheses=self.prev_hyp,
                drop_extra_pre_encoded=0 if first else int(self.scfg.drop_extra_pre_encoded or 0),
                return_transcription=True,
            )
            (_, transcribed, self.cache_ch, self.cache_t, self.cache_len, self.prev_hyp) = out
            last_text = self._hyp_text(transcribed)

            self.idx += shift
            self.step += 1
            if self.idx >= T:
                break
        return last_text

    @staticmethod
    def _hyp_text(transcribed) -> str | None:
        if not transcribed:
            return None
        first = transcribed[0]
        if isinstance(first, str):
            return first
        return getattr(first, "text", None) or ""
