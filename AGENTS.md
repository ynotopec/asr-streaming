# Audio-to-Audio Translator

Lowest-latency audio-to-audio translation pipeline.

- **Primary**: source-language speech → target-language speech (language1 → language2)
- **Secondary**: any supported language → primary language (others → primary)

## Stack

- **ASR**: `nvidia/nemotron-3.5-asr-streaming-0.6b` — multilingual streaming speech recognition (40 languages, cache-aware FastConformer-RNNT, 80ms–1120ms latency)
- **Translation**: MarianMT (`facebook/marian-mt-*`) — direct language pairs with English intermediary fallback
- **TTS**: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` — 10 languages, 9 voices, sub-100ms synthesis

## Function

```
translate(language1, language2)
```

Streaming WebSocket endpoint: real-time audio-in → audio-out.
Batch REST endpoint: upload audio file → receive translated audio.

## Streaming ASR (backend/streaming_asr.py)

- Cache-aware encoder: one encoder cache per utterance, fed contiguous feature chunks via `conformer_stream_step` (chunk/pre-encode/shift from `encoder.streaming_cfg`).
- Partial hypotheses pushed live; an utterance closes only after ≥0.6 s of silence (or 10 s max), then is translated and voiced.
- Do not transcribe isolated segments with `model.transcribe()` for the live path: it discards encoder context and degrades the output.
- Translation pairs must exist on HuggingFace. Verified: `opus-mt-ROMANCE-en` (pt→en), `opus-mt-tc-big-en-pt`, `opus-mt-en-fr`, `opus-mt-en-jap`, `opus-mt-tc-big-en-ko`, `opus-mt-tc-big-en-tr`. Absent pairs fall back to the English pivot.

## Languages

ASR supports 40 locales (transcription-ready, broad-coverage, adaptation-ready).
TTS supports 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian).
