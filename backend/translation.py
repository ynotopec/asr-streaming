"""
Translation engine: text→text using MarianMT with English pivot.
"""
import torch
from transformers import MarianMTModel, MarianTokenizer
from threading import Lock

SUPPORTED_LANGS = {
    "en-US": "en", "fr-FR": "fr", "de-DE": "de", "es-ES": "es",
    "it-IT": "it", "pt-BR": "pt", "zh-CN": "zh", "ja-JP": "ja",
    "ko-KR": "ko", "ru-RU": "ru", "ar-AR": "ar", "hi-IN": "hi",
    "nl-NL": "nl", "pl-PL": "pl", "tr-TR": "tr", "sv-SE": "sv",
    "cs-CZ": "cs", "nb-NO": "nb", "da-DK": "da", "fi-FI": "fi",
    "el-GR": "el", "th-TH": "th", "he-IL": "he", "uk-UA": "uk",
    "ro-RO": "ro", "hu-HU": "hu", "bg-BG": "bg", "hr-HR": "hr",
    "sk-SK": "sk", "lt-LT": "lt", "lv-LV": "lv", "et-EE": "et",
    "vi-VN": "vi",
}

_model_cache = {}
_lock = Lock()

# Helsinki-NLP Opus-MT pairs: {src→tgt: "Helsinki-NLP/opus-mt-xx-yy"}
_DIRECT_PAIRS = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
    ("en", "pt"): "Helsinki-NLP/opus-mt-tc-big-en-pt",
    ("pt", "en"): "Helsinki-NLP/opus-mt-ROMANCE-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
    ("en", "ko"): "Helsinki-NLP/opus-mt-tc-big-en-ko",
    ("ko", "en"): "Helsinki-NLP/opus-mt-ko-en",
    ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    ("fr", "de"): "Helsinki-NLP/opus-mt-fr-de",
    ("de", "fr"): "Helsinki-NLP/opus-mt-de-fr",
    ("es", "fr"): "Helsinki-NLP/opus-mt-es-fr",
    ("fr", "es"): "Helsinki-NLP/opus-mt-fr-es",
    ("it", "fr"): "Helsinki-NLP/opus-mt-it-fr",
    ("de", "es"): "Helsinki-NLP/opus-mt-de-es",
    ("es", "de"): "Helsinki-NLP/opus-mt-es-de",
    ("en", "nl"): "Helsinki-NLP/opus-mt-en-nl",
    ("nl", "en"): "Helsinki-NLP/opus-mt-nl-en",
    ("pl", "en"): "Helsinki-NLP/opus-mt-pl-en",
    ("tr", "en"): "Helsinki-NLP/opus-mt-tr-en",
    ("en", "tr"): "Helsinki-NLP/opus-mt-tc-big-en-tr",
    ("sv", "en"): "Helsinki-NLP/opus-mt-sv-en",
    ("en", "sv"): "Helsinki-NLP/opus-mt-en-sv",
    ("cs", "en"): "Helsinki-NLP/opus-mt-cs-en",
    ("en", "cs"): "Helsinki-NLP/opus-mt-en-cs",
    ("nb", "en"): "Helsinki-NLP/opus-mt-nb-en",
    ("en", "nb"): "Helsinki-NLP/opus-mt-en-nb",
    ("da", "en"): "Helsinki-NLP/opus-mt-da-en",
    ("en", "da"): "Helsinki-NLP/opus-mt-en-da",
    ("fi", "en"): "Helsinki-NLP/opus-mt-fi-en",
    ("en", "fi"): "Helsinki-NLP/opus-mt-en-fi",
}

PIVOT = "en"


def lang_code(locale_id: str) -> str:
    return SUPPORTED_LANGS.get(locale_id, locale_id.split("-")[0] if "-" in locale_id else locale_id)


def _load_translation_pair(src_lang: str, tgt_lang: str):
    src = lang_code(src_lang) if isinstance(src_lang, str) and "-" in src_lang else src_lang
    tgt = lang_code(tgt_lang) if isinstance(tgt_lang, str) and "-" in tgt_lang else tgt_lang

    if src == tgt:
        return None

    pair_key = (src, tgt)

    # Direct pair
    if pair_key in _DIRECT_PAIRS:
        cache_key = f"{src}-{tgt}"
        if cache_key not in _model_cache:
            with _lock:
                if cache_key not in _model_cache:
                    model_id = _DIRECT_PAIRS[pair_key]
                    tok = MarianTokenizer.from_pretrained(model_id)
                    mdl = MarianMTModel.from_pretrained(model_id, device_map="auto" if torch.cuda.is_available() else "cpu")
                    _model_cache[cache_key] = (tok, mdl)
        return _model_cache[cache_key]

    # English pivot fallback
    if src != PIVOT and src in SUPPORTED_LANGS.values():
        key1 = f"{src}-{PIVOT}"
        if key1 not in _model_cache:
            model_id = _DIRECT_PAIRS.get((src, PIVOT))
            if model_id:
                tok = MarianTokenizer.from_pretrained(model_id)
                mdl = MarianMTModel.from_pretrained(model_id, device_map="auto" if torch.cuda.is_available() else "cpu")
                _model_cache[key1] = (tok, mdl)
    if tgt != PIVOT and tgt in SUPPORTED_LANGS.values():
        key2 = f"{PIVOT}-{tgt}"
        if key2 not in _model_cache:
            model_id = _DIRECT_PAIRS.get((PIVOT, tgt))
            if model_id:
                tok = MarianTokenizer.from_pretrained(model_id)
                mdl = MarianMTModel.from_pretrained(model_id, device_map="auto" if torch.cuda.is_available() else "cpu")
                _model_cache[key2] = (tok, mdl)

    return ("pivot", src, tgt)


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    if not text:
        return text

    src = lang_code(src_lang) if isinstance(src_lang, str) and "-" in src_lang else src_lang
    tgt = lang_code(tgt_lang) if isinstance(tgt_lang, str) and "-" in tgt_lang else tgt_lang

    if src == tgt:
        return text

    pair = _load_translation_pair(src_lang, tgt_lang)
    if pair is None:
        return text

    if isinstance(pair, tuple) and len(pair) == 2:
        tok, model = pair
        device = model.device
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        return tok.decode(outputs[0], skip_special_tokens=True)

    if isinstance(pair, tuple) and pair[0] == "pivot":
        _, src_code, tgt_code = pair
        key1 = f"{src_code}-{PIVOT}"
        key2 = f"{PIVOT}-{tgt_code}"
        if key1 in _model_cache and key2 in _model_cache:
            tok1, mdl1 = _model_cache[key1]
            tok2, mdl2 = _model_cache[key2]
            device = mdl1.device
            inputs1 = tok1(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs1 = mdl1.generate(**inputs1)
            en_text = tok1.decode(outputs1[0], skip_special_tokens=True)
            inputs2 = tok2(en_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs2 = mdl2.generate(**inputs2)
            return tok2.decode(outputs2[0], skip_special_tokens=True)

    return text


def is_supported_pair(src_lang: str, tgt_lang: str) -> bool:
    src = lang_code(src_lang) if isinstance(src_lang, str) and "-" in src_lang else src_lang
    tgt = lang_code(tgt_lang) if isinstance(tgt_lang, str) and "-" in tgt_lang else tgt_lang
    if src == tgt:
        return True
    if (src, tgt) in _DIRECT_PAIRS:
        return True
    pivot_keys = (f"{src}-{PIVOT}", f"{PIVOT}-{tgt}")
    return all(k in _DIRECT_PAIRS for k in pivot_keys)


def init_translation():
    import logging
    logger = logging.getLogger(__name__)
    common_pairs = [
        ("fr-FR", "en-US"), ("en-US", "fr-FR"),
        ("en-US", "de-DE"), ("de-DE", "en-US"),
        ("en-US", "es-ES"), ("es-ES", "en-US"),
    ]
    for src, tgt in common_pairs:
        _load_translation_pair(src, tgt)
    logger.info(f"Translation pairs loaded: {len(_model_cache)}")