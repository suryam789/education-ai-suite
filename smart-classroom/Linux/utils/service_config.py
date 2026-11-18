"""Render settings for the edge-ai-libraries and performance-tools services."""

from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from utils.compose_profiles import resolve_profiles
from utils.registry_mode import parse_registry


ROOT = Path(__file__).resolve().parents[1]


ASR_DEFAULTS = {
    "provider": "openvino", "name": "whisper-small", "device": "CPU",
    "weight_format": None, "temperature": 0.0, "diarization": False,
    "models_base_path": "models", "hf_token": None,
    "vad_model": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc_model": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "no_speech_threshold": 0.3, "logprob_threshold": -0.3,
    "min_duration_sec": 0.5, "min_words": 2,
    "hallucination_silence_threshold": 1.0, "hallucination_phrase_filter": True,
    "hallucination_phrases": [
        "thank you", "thank you very much", "thanks for watching",
        "thank you for watching", "please subscribe", "subscribe", "you", "bye",
    ],
    "repetition_penalty": 1.8, "beam_size": 1, "best_of": 1,
    "word_timestamps": False, "threads": 0,
}
DIARIZATION_DEFAULTS = {
    "provider": "huggingface", "name": "pyannote/speaker-diarization-community-1",
    "device": "CPU", "models_base_path": "models", "min_speakers": 1,
    "max_speakers": 2,
    "identity": {
        "enabled": True, "similarity_threshold": 0.75,
        "lock_min_duration_sec": 0.75, "session_ttl_seconds": 1800,
    },
}
PREPROCESSING_DEFAULTS = {
    "chunk_duration_sec": 15, "silence_threshold": -35, "silence_duration": 0.3,
    "search_window_sec": 1, "chunk_output_path": "chunks/", "denoise": False,
    "denoise_model": "", "normalize": False, "normalize_filter": "dynaudnorm",
}
SENTIMENT_DEFAULTS = {
    "enabled": False, "model": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    "device": "CPU", "provider": "openvino", "weight_format": "int8",
    "models_base_path": "models", "recency_weight": 0.7, "peak_label": "angry",
}


def _selected(defaults: dict, settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("Configuration sections must be mappings")
    result = deepcopy(defaults)
    for key in defaults:
        if key in settings:
            if isinstance(defaults[key], dict):
                result[key] = _selected(defaults[key], settings[key])
            else:
                result[key] = deepcopy(settings[key])
    return result


def audio_config(data: dict[str, Any]) -> dict[str, Any]:
    """Produce a full service config without reading the microservice checkout."""
    models = data.get("models", {})
    source = models.get("asr", {})
    asr = _selected(ASR_DEFAULTS, source)
    if "threads" not in source:
        asr["threads"] = source.get("threads_limit") or 0
    asr["models_base_path"] = "models"
    asr["hf_token"] = None
    provider = asr["provider"]
    if provider not in {"openai", "openvino", "whispercpp", "funasr"}:
        raise ValueError(f"Unsupported models.asr.provider: {provider!r}")
    devices = {"CPU", "GPU", "NPU"} if provider == "openvino" else {"CPU"}
    if asr["device"] not in devices:
        raise ValueError(f"models.asr.device must be one of {sorted(devices)} for {provider}")
    if not isinstance(asr["name"], str) or not asr["name"].strip():
        raise ValueError("models.asr.name must be a non-empty string")
    if provider == "funasr" and asr["name"] not in {
        "paraformer-zh", "paraformer-en", "paraformer-online",
    }:
        raise ValueError(f"Unsupported FunASR model: {asr['name']!r}")
    if asr["diarization"] and provider != "openvino":
        raise ValueError("models.asr.diarization requires provider: openvino")
    if "preview" in source:
        preview = source["preview"]
        if preview.get("provider", provider) != provider:
            raise ValueError("models.asr.preview must use the main ASR provider")
        if preview.get("device", asr["device"]) not in devices:
            raise ValueError("models.asr.preview.device is not supported by the provider")
        asr["preview"] = {
            key: value for key, value in preview.items()
            if key in {"name", "device", "weight_format"}
        }
    diarization = models.get("diarization", {})
    if asr["diarization"] and diarization.get("backend", "pyannote") != "pyannote":
        raise ValueError("Audio Analyzer supports models.diarization.backend: pyannote only")
    service = data.get("audio_analyzer", {})
    return {
        "app": {
            "name": "Audio Analyzer", "cleanup_on_exit": service.get("cleanup_on_exit", True),
            "clear_storage_on_startup": service.get("clear_storage_on_startup", False),
            "use_ov_genai": service.get("use_ov_genai", True),
        },
        "models": {
            "asr": asr,
            "diarization": _selected(DIARIZATION_DEFAULTS, diarization),
        },
        "audio_preprocessing": _selected(PREPROCESSING_DEFAULTS, data.get("audio_preprocessing", {})),
        "audio_util": _selected({
            "max_size_mb": 200,
            "allowed_extensions": [".wav", ".mp3", ".m4a", ".mp4", ".mkv", ".mov", ".avi"],
            "chunk_size": 52428800,
        }, data.get("audio_util", {})),
        "pipeline": {"delete_chunks_after_use": data.get("pipeline", {}).get("delete_chunks_after_use", True)},
        "minio": {"endpoint": "", "access_key": "", "secret_key": "", "secure": False},
        "sentiment": _selected(SENTIMENT_DEFAULTS, service.get("sentiment", {})),
    }


def _profiles(data: dict) -> list[str]:
    features = data.get("features")
    settings = SimpleNamespace()
    if features is not None:
        if not isinstance(features, dict):
            raise ValueError("features must be a mapping")
        settings.features = SimpleNamespace(**{
            key: SimpleNamespace(**value) for key, value in features.items()
        })
    return resolve_profiles(settings)


def compose_environment(data: dict, output: Path) -> dict:
    deployment = data.get("deployment", {})
    images = deployment.get("images", {})
    downloader = data.get("model_download", {})
    profiles = _profiles(data)
    plugins = downloader.get("plugins", ["huggingface", "openvino", "modelscope"])
    if not isinstance(plugins, list) or not plugins or any(
        plugin not in {"huggingface", "openvino", "modelscope", "ollama"} for plugin in plugins
    ):
        raise ValueError("model_download.plugins must be a list of supported plugin names")
    asr = data.get("models", {}).get("asr", {})
    if "asr" in profiles and asr.get("provider") == "funasr" and "modelscope" not in plugins:
        raise ValueError("FunASR requires modelscope in model_download.plugins")
    if "text-gen" in profiles and "openvino" not in plugins:
        raise ValueError("OVMS exports require openvino in model_download.plugins")
    return {
        "SC_GENERATED_DIR": str(output.resolve()),
        "COMPOSE_PROFILES": ",".join(profiles),
        "STARTUP_MODELS_CONFIG": "/opt/startup-models.yaml" if "text-gen" in profiles else "",
        "AUDIO_ANALYZER_IMAGE": images.get("audio_analyzer", "intel/audio-analyzer:2026.3.0-rc1"),
        "MODEL_DOWNLOADER_IMAGE": images.get("model_downloader", "intel/model-download:latest"),
        "METRICS_COLLECTOR_IMAGE": images.get("metrics_collector", "intel/retail-benchmark:2026.3.0-rc1"),
        "OVMS_RELEASE_TAG": downloader.get("ovms_release_tag", "v2026.4.1"),
        "ENABLED_PLUGINS": ",".join(plugins),
        "INSTALL_CJK": asr.get("provider") == "funasr",
        "AUDIO_ANALYZER_MODEL_DOWNLOAD_TIMEOUT_SEC": data.get("audio_analyzer", {}).get("model_download_timeout_sec", 1800),
    }


def _env_text(values: dict) -> str:
    lines = []
    for key, value in values.items():
        text = str(value).lower() if isinstance(value, bool) else str(value)
        if "\n" in text or "\r" in text:
            raise ValueError(f"Multiline deployment value is not supported: {key}")
        text = text.replace("'", "\\'")
        lines.append(f"{key}='{text}'")
    return "\n".join(lines) + "\n"


def generate(data: dict, output: Path) -> dict:
    """Validate and materialize service inputs; never consult submodule files."""
    parse_registry(data.get("REGISTRY"))
    profiles = _profiles(data)
    environment = compose_environment(data, output)
    artifacts = {}
    if "asr" in profiles:
        artifacts["audio-analyzer.yaml"] = audio_config(data)
    else:
        artifacts["audio-analyzer.yaml"] = audio_config({})
    rendered = {
        name: yaml.safe_dump(value, sort_keys=False) for name, value in artifacts.items()
    }
    environment["SC_AUDIO_CONFIG_SHA256"] = hashlib.sha256(
        rendered["audio-analyzer.yaml"].encode("utf-8")
    ).hexdigest()
    rendered["compose.env"] = _env_text(environment)
    output.mkdir(parents=True, exist_ok=True)
    for name, text in rendered.items():
        path = output / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("SC_CONFIG_PATH", ROOT / "config.yaml")))
    parser.add_argument("--output", type=Path, default=ROOT / ".generated")
    args = parser.parse_args()
    os.environ["SC_CONFIG_PATH"] = str(args.config.resolve())
    from utils.config_loader import _validate_features, _validate_text_gen
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        parser.error("The Smart Classroom configuration must be a YAML mapping")
    _validate_features(data)
    _validate_text_gen(data)
    generate(data, args.output)
    print(f"Generated service configuration in {args.output.resolve()}")


if __name__ == "__main__":
    main()