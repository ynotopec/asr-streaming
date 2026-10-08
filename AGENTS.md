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

## Languages

ASR supports 40 locales (transcription-ready, broad-coverage, adaptation-ready).
TTS supports 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian).
