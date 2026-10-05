"""
Speaker diarization for the streaming ASR app.

Nemotron 3.5 ASR is a pure transcription model (no speaker output), so speaker
identification is provided by a separate, language-agnostic pipeline:

    audio → Silero VAD → ECAPA-TDNN speaker embeddings → clustering → timeline

Design choices
--------------
* **ECAPA-TDNN embeddings** (`speechbrain/spkrec-ecapa-voxceleb`): 192-d
  x-vectors. Language-agnostic — discriminate French speakers even though the
  checkpoint was trained on English VoxCeleb. Measured on a mixed test:
  same-speaker cosine ≈ 0.52, cross-speaker ≈ -0.04 → very clean separation
  (WavLM-base-SV gave much weaker separation and was rejected).
* **Silero VAD**: gates out silence so only speech windows are embedded.
* **Runs on CPU** (~50 ms per 1.5 s window): leaves the GPU entirely to the ASR
  model, so diarization never competes for VRAM or slows transcription.
* **Two-stage clustering**: a lightweight online pass gives real-time speaker
  tracking; `finalize()` re-clusters every embedding agglomeratively for a
  refined result emitted with the final transcript.

The diarizer runs in its own thread fed by a queue, so embedding never blocks
the ASR generation thread or the WebSocket event loop.
"""

import logging
import queue
import threading
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# ── Model handles (loaded lazily, shared process-wide) ──────────
_embedder = None
_vad = None
_models_lock = threading.Lock()


class SpeakerEmbedder:
    """ECAPA-TDNN speaker embedding extractor (speechbrain)."""

    def __init__(self, device: str = "cpu"):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )

    @torch.no_grad()
    def embed(self, wav: np.ndarray) -> np.ndarray:
        """Return an L2-normalised embedding for a 16 kHz mono buffer."""
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
        if self.device != "cpu":
            x = x.to(self.device)
        v = self.model.encode_batch(x).squeeze().cpu().numpy()
        v = np.atleast_1d(v).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class SileroVAD:
    """Thin wrapper around Silero VAD (16 kHz)."""

    def __init__(self):
        self.model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True, verbose=False
        )
        self.model.eval()
        self.frame = 512  # 32 ms @ 16 kHz

    @torch.no_grad()
    def speech_prob(self, wav: np.ndarray) -> float:
        self.model.reset_states()
        probs = []
        n = len(wav) - (len(wav) % self.frame)
        for i in range(0, n, self.frame):
            frame = torch.from_numpy(wav[i : i + self.frame]).float()
            probs.append(float(self.model(frame, SAMPLE_RATE).item()))
        return max(probs) if probs else 0.0


def get_models(device: str = "cpu"):
    """Lazily build and cache the embedder + VAD (thread-safe)."""
    global _embedder, _vad
    if _embedder is None or _vad is None:
        with _models_lock:
            if _embedder is None:
                logger.info("Loading speaker embedder (ECAPA-TDNN)…")
                _embedder = SpeakerEmbedder(device=device)
                logger.info("Speaker embedder ready ✓")
            if _vad is None:
                logger.info("Loading Silero VAD…")
                _vad = SileroVAD()
                logger.info("Silero VAD ready ✓")
    return _embedder, _vad


class OnlineDiarizer:
    """
    Streaming diarizer: consumes PCM chunks, maintains a speaker timeline.

    `speaker_at(t)` answers "who was speaking at audio-second t".
    `finalize()` refines the timeline with agglomerative clustering.
    """

    def __init__(
        self,
        max_speakers: int = 4,
        sim_threshold: float = 0.30,
        window_s: float = 1.5,
        hop_s: float = 0.5,
        min_speech_prob: float = 0.5,
        patience: int = 2,
        device: str = "cpu",
    ):
        self.embedder, self.vad = get_models(device=device)
        self.max_speakers = max_speakers
        self.sim_threshold = sim_threshold
        self.win_len = int(window_s * SAMPLE_RATE)
        self.hop_len = int(hop_s * SAMPLE_RATE)
        self.min_speech_prob = min_speech_prob
        self.patience = patience

        self._q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

        # clustering state
        self._centroids: List[np.ndarray] = []
        self._counts: List[int] = []
        self._pending_emb: Optional[np.ndarray] = None
        self._pending_n = 0
        self._lock = threading.Lock()

        # history for refinement
        self._embeddings: List[Tuple[float, np.ndarray]] = []
        # timeline: list of (t_end_seconds, speaker_id)
        self._points: List[Tuple[float, int]] = []
        self._current: Optional[int] = None

    # ── lifecycle ────────────────────────────────────────────
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, chunk: np.ndarray):
        self._q.put(chunk.astype(np.float32, copy=False))

    def close(self):
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ── worker ───────────────────────────────────────────────
    def _run(self):
        buf = np.zeros(0, dtype=np.float32)
        buf_start = 0
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                if item.size == 0:
                    continue
                buf = np.concatenate([buf, item]) if buf.size else item
                while buf.shape[0] >= self.hop_len:
                    win = buf[-self.win_len :] if buf.shape[0] >= self.win_len else buf
                    t_end = (buf_start + buf.shape[0]) / SAMPLE_RATE
                    self._process_window(win, t_end)
                    buf = buf[self.hop_len :]
                    buf_start += self.hop_len
        except Exception as e:
            logger.error(f"Diarizer thread error: {e}")

    def _process_window(self, win: np.ndarray, t_end: float):
        if win.shape[0] < int(0.25 * SAMPLE_RATE):
            return
        try:
            if self.vad.speech_prob(win) < self.min_speech_prob:
                return  # silence: keep the previous speaker as "current"
        except Exception:
            pass
        try:
            emb = self.embedder.embed(win)
        except Exception as e:
            logger.error(f"Embed error: {e}")
            return
        spk = self._assign(emb)
        with self._lock:
            self._embeddings.append((t_end, emb))
            self._points.append((t_end, spk))
            self._current = spk

    def _assign(self, emb: np.ndarray) -> int:
        """Online assignment: nearest centroid, or a new speaker after `patience`."""
        with self._lock:
            if not self._centroids:
                self._centroids.append(emb)
                self._counts.append(1)
                self._pending_emb, self._pending_n = None, 0
                return 0
            sims = [float(np.dot(emb, c)) for c in self._centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self.sim_threshold:
                self._update_centroid(best, emb)
                self._pending_emb, self._pending_n = None, 0
                return best
            # below threshold: do not create a cluster on a single outlier
            if len(self._centroids) < self.max_speakers:
                if (
                    self._pending_emb is not None
                    and float(np.dot(emb, self._pending_emb)) >= self.sim_threshold
                ):
                    self._pending_n += 1
                else:
                    self._pending_emb, self._pending_n = emb, 1
                if self._pending_n >= self.patience:
                    self._centroids.append(emb)
                    self._counts.append(1)
                    self._pending_emb, self._pending_n = None, 0
                    return len(self._centroids) - 1
            return best  # transitional: stay with the nearest speaker

    def _update_centroid(self, idx: int, emb: np.ndarray):
        c, n = self._centroids[idx], self._counts[idx]
        new = (c * n + emb) / (n + 1)
        nn = np.linalg.norm(new)
        self._centroids[idx] = new / nn if nn > 0 else new
        self._counts[idx] = n + 1

    # ── refinement ───────────────────────────────────────────
    def finalize(self):
        """Re-cluster all embeddings globally and rebuild the timeline."""
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        with self._lock:
            embs = [e for _, e in self._embeddings]
            times = [t for t, _ in self._embeddings]
            if len(embs) < 4:
                return
            X = np.array(embs)
            kmax = min(self.max_speakers, len(X) - 1)
            min_size = max(3, int(0.12 * len(X)))
            # choose k by silhouette (cosine), but only accept a split that is
            # (a) balanced enough and (b) genuinely separated. Otherwise the
            # audio is treated as single-speaker.
            best_k, best_score = 1, -2.0
            for k in range(2, kmax + 1):
                try:
                    cl = AgglomerativeClustering(
                        n_clusters=k, metric="cosine", linkage="average"
                    ).fit(X)
                    labels_k = cl.labels_
                    if len(set(labels_k)) < 2:
                        continue
                    sizes = np.bincount(labels_k)
                    if sizes[sizes > 0].min() < min_size:
                        continue  # reject lopsided split (isolated outlier)
                    cents = np.array(
                        [X[labels_k == c].mean(0) for c in range(k) if (labels_k == c).any()]
                    )
                    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
                    sim = cents @ cents.T
                    off = sim[~np.eye(len(cents), dtype=bool)]
                    if off.max() > 0.50:
                        continue  # centroids too close → same speaker
                    s = silhouette_score(X, labels_k, metric="cosine")
                    if s > best_score:
                        best_k, best_score = k, s
                except Exception:
                    continue
            if best_k == 1:
                self._points = [(t, 0) for t in times]
                self._current = 0
                logger.info("Diarization refined → 1 speaker (silhouette check found no split)")
                return
            cl = AgglomerativeClustering(
                n_clusters=best_k, metric="cosine", linkage="average"
            ).fit(X)
            labels = cl.labels_
            # renumber speakers by first appearance
            order = {}
            for lab in labels:
                if lab not in order:
                    order[lab] = len(order)
            self._points = [(t, order[lab]) for t, lab in zip(times, labels)]
            self._current = self._points[-1][1] if self._points else None
            logger.info(f"Diarization refined → {best_k} speaker(s), silhouette={best_score:.3f}")

    # ── queries ──────────────────────────────────────────────
    def speaker_at(self, t: float, tol: float = 1.5) -> Optional[int]:
        """Speaker active at audio-second `t` (nearest recorded point within tol)."""
        with self._lock:
            best = None
            for pt_t, spk in self._points:
                if pt_t <= t + tol:
                    best = spk
                else:
                    break
            return best if best is not None else self._current

    @property
    def current_speaker(self) -> Optional[int]:
        with self._lock:
            return self._current

    @property
    def n_speakers(self) -> int:
        with self._lock:
            return len({s for _, s in self._points}) or len(self._centroids)
