from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from utils.service_config import audio_config, generate


ROOT = Path(__file__).resolve().parents[2]


def _isolated_checkout(tmp_path):
    for folder in ("utils", "scripts", "docker"):
        (tmp_path / folder).mkdir()
    for name in ("service_config.py", "compose_profiles.py", "registry_mode.py", "config_loader.py"):
        shutil.copyfile(ROOT / "utils" / name, tmp_path / "utils" / name)
    shutil.copyfile(ROOT / "setup.sh", tmp_path / "setup.sh")
    shutil.copyfile(ROOT / "scripts/compose.sh", tmp_path / "scripts/compose.sh")
    for name in ("docker-compose.yml", "docker-compose.gpu.yml", "docker-compose.local-build.yml"):
        shutil.copyfile(ROOT / "docker" / name, tmp_path / "docker" / name)


def test_audio_config_maps_user_settings_without_submodule(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = {
        "models": {"asr": {
            "provider": "funasr", "name": "paraformer-en", "device": "CPU",
            "threads_limit": 4, "concurrency": 8, "queue_max": 16,
        }},
        "audio_preprocessing": {"chunking": True, "chunk_duration_sec": 30},
        "audio_analyzer": {"use_ov_genai": False},
    }
    original = deepcopy(data)
    result = audio_config(data)
    assert result["models"]["asr"]["name"] == "paraformer-en"
    assert result["models"]["asr"]["threads"] == 4
    assert "concurrency" not in result["models"]["asr"]
    assert "queue_max" not in result["models"]["asr"]
    assert "chunking" not in result["audio_preprocessing"]
    assert result["audio_preprocessing"]["chunk_duration_sec"] == 30
    assert result["app"]["use_ov_genai"] is False
    assert result["sentiment"]["enabled"] is False
    assert data == original


@pytest.mark.parametrize("settings", [
    {"provider": "invalid"},
    {"provider": "funasr", "name": "paraformer-zh", "device": "GPU"},
    {"provider": "funasr", "name": "unknown"},
    {"provider": "openai", "diarization": True},
    {"name": ""},
    {"preview": {"provider": "openai"}},
])
def test_audio_config_rejects_unsupported_settings(settings):
    with pytest.raises(ValueError):
        audio_config({"models": {"asr": settings}})


def test_audio_config_keeps_tokens_out_of_generated_files():
    result = audio_config({"models": {"asr": {"hf_token": "secret"}}})
    assert result["models"]["asr"]["hf_token"] is None


def test_audio_config_preserves_nested_defaults():
    result = audio_config({"models": {"diarization": {"identity": {"enabled": False}}}})
    assert result["models"]["diarization"]["identity"]["enabled"] is False
    assert result["models"]["diarization"]["identity"]["session_ttl_seconds"] == 1800


def test_generate_registry_configuration_without_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = {
        "REGISTRY": True, "features": {"asr": {"enabled": True}},
        "models": {"asr": {"provider": "funasr", "name": "paraformer-en"}},
        "deployment": {"images": {"audio_analyzer": "example/audio:release"}},
    }
    output = tmp_path / "generated"
    environment = generate(data, output)
    assert environment["AUDIO_ANALYZER_IMAGE"] == "example/audio:release"
    assert environment["COMPOSE_PROFILES"] == "asr"
    assert environment["STARTUP_MODELS_CONFIG"] == ""
    assert environment["INSTALL_CJK"] is True
    assert yaml.safe_load((output / "audio-analyzer.yaml").read_text())["models"]["asr"]["name"] == "paraformer-en"
    original = (output / "compose.env").read_text()
    generate(data, output)
    assert (output / "compose.env").read_text() == original
    assert not (tmp_path / "edge-ai-libraries").exists()


def test_generation_is_limited_to_submodule_services(tmp_path):
    environment = generate({}, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {"audio-analyzer.yaml", "compose.env"}
    assert not {"SC_CONFIG_FILE", "APP_PORT", "UI_PORT", "OVMS_IMAGE", "STORAGE_DIR"} & environment.keys()
    assert not any(key.startswith(("VA_", "VITE_")) for key in environment)


def test_disabled_asr_does_not_validate_unused_provider(tmp_path):
    data = {"features": {}, "models": {"asr": {"provider": "unused"}}}
    assert generate(data, tmp_path / "output")["COMPOSE_PROFILES"] == ""


def test_failed_validation_does_not_replace_generated_configuration(tmp_path):
    output = tmp_path / "generated"
    generate({}, output)
    before = (output / "compose.env").read_text()
    with pytest.raises(ValueError, match="modelscope"):
        generate({
            "features": {"asr": {"enabled": True}},
            "models": {"asr": {"provider": "funasr", "name": "paraformer-zh"}},
            "model_download": {"plugins": ["huggingface"]},
        }, output)
    assert (output / "compose.env").read_text() == before


@pytest.mark.parametrize("registry", [True, False])
def test_compose_uses_selected_config_without_submodules(tmp_path, registry):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose is not installed")
    _isolated_checkout(tmp_path)
    source = tmp_path / "selected.yaml"
    data = {
        "REGISTRY": registry, "features": {"asr": {"enabled": True}},
        "models": {"asr": {"provider": "funasr", "name": "paraformer-en"}},
        "deployment": {"images": {
            "audio_analyzer": "example/audio:release",
            "model_downloader": "example/downloader:release",
            "metrics_collector": "example/metrics:release",
        }},
    }
    source.write_text(yaml.safe_dump(data))
    (tmp_path / ".env").write_text("RENDER_GID=109\nAUDIO_ANALYZER_IMAGE=stale/image:tag\nSTARTUP_MODELS_CONFIG=/stale.yaml\n")
    environment = {
        **os.environ, "SC_CONFIG_PATH": str(source), "AUDIO_ANALYZER_IMAGE": "stale/shell:tag",
        "RENDER_GID": "109",
        "SERVING_BASE_URL": "http://custom-ovms:9000",
        "COMPOSE_PROFILES": "text-gen", "STARTUP_MODELS_CONFIG": "/stale-shell.yaml",
    }
    command = ["bash", str(tmp_path / "scripts/compose.sh"), "config", "--format", "json"]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    services = json.loads(result.stdout)["services"]
    analyzer = services["audio-analyzer"]
    assert analyzer["image"] == ("example/audio:release" if registry else "smart-classroom-audio-analyzer:local")
    assert ("build" in analyzer) is not registry
    assert services["model-downloader"]["image"] == ("example/downloader:release" if registry else "sc-model-downloader:local")
    assert services["metrics-collector"]["image"] == ("example/metrics:release" if registry else "sc-metrics-collector:local")
    assert services["app"]["environment"]["SERVING_BASE_URL"] == "http://custom-ovms:9000"
    assert "SC_CONFIG_PATH" not in services["app"]["environment"]
    assert services["model-downloader"]["environment"].get("STARTUP_MODELS_CONFIG") in (None, "")
    assert "ovms" not in services
    mounts = analyzer["volumes"]
    assert not any("edge-ai-libraries" in volume["source"] for volume in mounts)
    assert not (tmp_path / "edge-ai-libraries").exists()
    data["models"]["asr"]["name"] = "paraformer-zh"
    source.write_text(yaml.safe_dump(data))
    updated = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
    updated_analyzer = json.loads(updated.stdout)["services"]["audio-analyzer"]
    assert updated_analyzer["environment"]["SC_AUDIO_CONFIG_SHA256"] != analyzer["environment"]["SC_AUDIO_CONFIG_SHA256"]
    generated = yaml.safe_load((tmp_path / ".generated/audio-analyzer.yaml").read_text())
    assert generated["models"]["asr"]["name"] == "paraformer-zh"


@pytest.mark.parametrize("tag_override", [None, "custom-release"])
def test_fresh_registry_install_uses_upstream_image_tag(tmp_path, tag_override):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose is not installed")
    _isolated_checkout(tmp_path)
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump({
        "REGISTRY": True,
        "features": {"summary": {"enabled": True}, "video_analytics": {"enabled": True}},
    }))
    shutil.copyfile(ROOT / ".env.example", tmp_path / ".env")
    environment = {**os.environ, "SC_CONFIG_PATH": str(source), "RENDER_GID": "109"}
    for key in ("INTEL_IMAGE_TAG", "SMART_CLASSROOM_APP_IMAGE", "OVMS_PROVISION_IMAGE",
                "VA_MODEL_PROVISION_IMAGE", "VIDEO_ANALYTICS_IMAGE"):
        environment.pop(key, None)
    if tag_override is not None:
        environment["INTEL_IMAGE_TAG"] = tag_override
    result = subprocess.run(
        ["bash", str(tmp_path / "scripts/compose.sh"), "config", "--format", "json"],
        env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    services = json.loads(result.stdout)["services"]
    tag = tag_override or "2026.3.0-rc1"
    for service, repository in {
        "app": "smart-classroom-app",
        "ovms-provision": "smart-classroom-ovms-provision",
        "va-model-provision": "smart-classroom-va-model-provision",
        "video-analytics": "smart-classroom-video-analytics",
    }.items():
        assert services[service]["image"] == f"intel/{repository}:{tag}"
        assert "build" not in services[service]


@pytest.mark.parametrize("registry", [True, False])
def test_setup_initializes_submodules_only_for_local_build(tmp_path, registry):
    _isolated_checkout(tmp_path)
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump({"REGISTRY": registry, "features": {}}))
    (tmp_path / "scripts/node-setup.sh").write_text("exit 23\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$GIT_CALLS"\n')
    git.chmod(0o755)
    calls = tmp_path / "git-calls"
    result = subprocess.run(
        ["bash", str(tmp_path / "setup.sh")], capture_output=True, text=True,
        env={**os.environ, "SC_CONFIG_PATH": str(source), "GIT_CALLS": str(calls),
             "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"]},
    )
    assert result.returncode == 23, result.stderr
    if registry:
        assert not calls.exists()
    else:
        commands = calls.read_text()
        assert "submodule update --init edge-ai-libraries" in commands
        assert "submodule update --init performance-tools" in commands
