# Smart Classroom

The **Smart Classroom** project is a modular, extensible framework designed to process and summarize educational content using advanced AI models for the Linux OS. It supports transcription, summarization, mindmap generation, future capabilities like video understanding, multimodal content search, and real-time video analytics.

## Main Features

The main features are as follows:

### Audio Pipeline

- Audio transcription with ASR models (e.g., Whisper, Paraformer)
- Speaker diarization using Pyannote Audio models (optional)
- Summarization using powerful LLMs (e.g., Qwen, LLaMA)
- MindMap generation using Mermaid.js for visual diagram rendering of the summary
- Content segmentation for automatic topic extraction from transcripts
- Timestamped topic indexing and semantic search across classroom sessions

### Video Analytics Pipeline

- Real-time video analysis across three concurrent camera streams (front, back, content)
- Person detection and pose estimation using YOLO models with keypoint skeleton tracking
- Person re-identification for tracking individual students across frames
- Classroom statistics including student count, stand-up events, and hand-raise events

### Content Search

- Multimodal semantic search across videos, images, and documents
- AI-driven video summarization using Vision Language Models (VLM)
- Document text extraction with OCR support (PaddleOCR)
- RAG-based Question & Answer over uploaded educational materials
- Vector-based retrieval using ChromaDB with CLIP and mE5 embeddings

Topic segmentation automatically enables Content Search.

- The reranker requires an Intel GPU.
- Run `./setup.sh` to configure GPU access and start the app.

For manual startup and configuration changes, use the commands below.

## Configuration

[config.yaml](config.yaml) controls the services supplied by `edge-ai-libraries`
and `performance-tools`. Python 3 and PyYAML are required on the host.
`REGISTRY: true` skips cloning both submodules and uses published audio analyzer,
model downloader, and metrics collector images. `REGISTRY: false` initializes
both submodules and builds those images locally.

- `models.asr`: audio provider, model, device, decoder/filter settings and optional
	`preview` model settings. CPU is required for FunASR, OpenAI and whisper.cpp;
	OpenVINO supports CPU/GPU/NPU. Diarization requires OpenVINO and pyannote.
- `models.diarization`, `audio_preprocessing`, `audio_util`, `pipeline`, and
	`audio_analyzer`: diarization, audio processing, service lifecycle and sentiment.
	`audio_analyzer.host_addr`/`port` are native-client settings; Docker uses service DNS.
- `deployment.images`: image references for these three services only.
- `model_download`: downloader plugins and the OVMS exporter release. FunASR
	requires the ModelScope plugin. The existing startup-manifest synchronization
	for text-generation models is unchanged.

Run the initial installation with `./setup.sh`. After editing audio settings:

```bash
bash scripts/compose.sh config --quiet
bash scripts/compose.sh up -d audio-analyzer
```

Use `bash scripts/compose.sh` in place of `docker compose` for `logs`, `ps`, `pull`,
and other stack commands. It selects the registry/local-build files automatically,
generates the audio configuration, and prevents old environment values for these
three services from overriding the main config. For local builds after adding
FunASR, run `bash scripts/compose.sh up -d --build audio-analyzer` to include its
dependencies. An audio-config fingerprint makes Compose recreate audio analyzer
when settings change; unchanged settings do not force recreation of the stack.

The ignored `.generated/` directory contains only the audio analyzer config and
the Compose environment values for these services. Do not edit these generated
outputs or the audio analyzer configuration inside the submodule checkout.
`python3 -m utils.service_config` generates files without starting containers.

Secrets such as `HF_TOKEN`, proxies, corporate CA paths and the detected render group
remain in the ignored `.env` or host environment. Other application, UI, video
analytics, Content Search, storage, and project-state configuration is unchanged.
The main app, OVMS provisioner, and video analytics services also use published
images in registry mode; their local builds are selected with `REGISTRY: false`.

The main config's existing thresholds, chunk duration and upload limits now apply to
audio analyzer; they previously could differ from the submodule's configuration.
Advanced defaults preserve the audio analyzer filter and sentiment defaults. Migrate
any custom submodule settings into the corresponding main-config sections before restarting.

Published images must contain the selected provider and downloader plugins. The
`intel/model-download:2026.3.0-rc1` release does not provide ModelScope; the configured
`latest` is retained for compatibility. Pin a tested ModelScope-capable release or
digest for production. Selecting a provider cannot add missing dependencies to an image.

## Architecture

- Plug-and-play architecture for integrating new ASR, LLM, and video analytics models
- API-first design ready for frontend integration.

### OVMS pipeline configuration

`models.text_gen.ovms.pipeline_type` defaults to `VLM`. Setup copies it into
`config.pipeline_type` in both startup-models configurations; the model downloader
passes it to `export_model.py text_generation --pipeline_type`. Generating a
startup configuration with `python -m utils.ovms_provision --write-startup-config`
preserves the same setting. Valid values are `LM`, `LM_CB`, `VLM`, `VLM_CB` and
`AUTO`.

Run `./setup.sh` after changing this setting. It controls model export/serving
configuration, not individual chat requests; editing it does not change an
already-running OVMS pipeline.

## Get Started

To see the system requirements and other installations, see the following guides:

- [System Requirements](../docs/user-guide/get-started/system-requirements.md): Check the hardware and software requirements for deploying the application.
- [Get Started](../docs/user-guide/get-started/get-started-linux.md): Follow step-by-step instructions to set up the application.
- [Uninstall](../docs/user-guide/get-started/get-started-linux.md#uninstall-the-application): Run `./uninstall.sh` to remove the containers, models, and data.
- [Linux Docker Architecture](../docs/user-guide/linux-docker-architecture.md): See the containers, ports, startup order, and data flow in the Docker Compose stack.
