# Nekro Agent AI Image Plugin

A Nekro Agent plugin for generating and editing images, supporting multiple backends **simultaneously**:

- **OpenAI** (GPT Image, DALL·E, or any OpenAI-compatible endpoint)
- **Gemini** (Google Gemini with native image generation)
- **Stable Diffusion** (A1111 / Forge WebUI API)

## Features

- **Multi-backend simultaneous configuration** — configure all three backends at once, switch via commands
- **Slash commands**: `/na-gpt`, `/na-gemini`, `/na-sd` for direct image generation
- **Preset system** — save image presets (for quick img2img) and text presets (prompt templates)
- **WebUI API** — upload/manage presets from the Nekro Agent WebUI
- **Sandbox tools** — AI can auto-call image generation, editing, and preset-based generation
- Text-to-image generation
- Single-image editing
- Multi-image reference editing

## Installation

Copy this plugin directory into your Nekro Agent plugin packages directory:

```bash
plugins/packages/gpt_image
```

Then restart Nekro Agent or reload plugins from the plugin manager.

## Configuration

Each backend has its own model group, so you can configure all three simultaneously.

### Backend Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `DEFAULT_BACKEND` | Default backend for AI auto-calls: `openai`, `gemini`, or `sd_webui` | `openai` |
| `OPENAI_MODEL_GROUP` | Model group for OpenAI backend | `""` |
| `OPENAI_MODEL_NAME` | OpenAI model name | `gpt-image-1` |
| `GEMINI_MODEL_GROUP` | Model group for Gemini backend | `""` |
| `GEMINI_MODEL_NAME` | Gemini model name | `gemini-2.0-flash-preview-image-generation` |
| `SD_MODEL_GROUP` | Model group for SD WebUI backend | `""` |
| `SD_MODEL_NAME` | SD checkpoint override (empty = use current) | `""` |

### Common Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `DEFAULT_SIZE` | Image size: `auto` or `WIDTHxHEIGHT` | `1024x1024` |
| `OUTPUT_FORMAT` | Output format: `png`, `jpeg`, or `webp` | `png` |
| `OUTPUT_COMPRESSION` | JPEG/WebP quality (1–100) | `100` |
| `TIMEOUT_SECONDS` | HTTP timeout | `300` |
| `MAX_REFERENCE_IMAGES` | Max reference images for multi-edit | `5` |

### OpenAI-only Settings

| Setting | Default |
|---------|---------|
| `DEFAULT_QUALITY` | `auto` |
| `DEFAULT_BACKGROUND` | `auto` |
| `MODERATION` | `auto` |

### Stable Diffusion Settings

| Setting | Default |
|---------|---------|
| `SD_NEGATIVE_PROMPT` | `""` |
| `SD_STEPS` | `20` |
| `SD_CFG_SCALE` | `7.0` |
| `SD_SAMPLER` | `Euler a` |
| `SD_DENOISING_STRENGTH` | `0.75` |

### Example Model Group Configurations

**OpenAI / compatible API:**
```yaml
MODEL_GROUPS:
  openai_image:
    BASE_URL: "https://api.openai.com/v1"
    API_KEY: "${OPENAI_API_KEY}"
    CHAT_MODEL: "gpt-image-1"
```

**Google Gemini:**
```yaml
MODEL_GROUPS:
  gemini_image:
    BASE_URL: "https://generativelanguage.googleapis.com/v1beta"
    API_KEY: "${GEMINI_API_KEY}"
    CHAT_MODEL: "gemini-2.0-flash-preview-image-generation"
```

**Stable Diffusion WebUI (local):**
```yaml
MODEL_GROUPS:
  sd_local:
    BASE_URL: "http://127.0.0.1:7860"
    API_KEY: ""
```

## Slash Commands

| Command | Description |
|---------|-------------|
| `/na-gpt <prompt>` | Generate image using OpenAI backend |
| `/na-gemini <prompt>` | Generate image using Gemini backend |
| `/na-sd <prompt>` | Generate image using SD WebUI backend |
| `/na-preset.list` | List all saved presets |
| `/na-preset.add-text <name> <template>` | Add a text preset (use `{input}` as placeholder) |
| `/na-preset.delete <name>` | Delete a preset |

Aliases: `gpt画图`, `gemini画图`, `sd画图`

## Preset System

### Text Presets

Text presets are prompt templates with an optional `{input}` placeholder:

```
/na-preset.add-text 动漫风 将以下内容画成动漫风格: {input}
```

Then use it: `/na-gpt 动漫风` (if the prompt matches a preset name, the template is used).

### Image Presets

Image presets store a reference image for quick img2img. Upload via the WebUI API:

```
POST /plugins/{plugin_key}/presets/image
  - name: preset name
  - file: image file upload
  - description: optional description
  - default_prompt: optional default prompt
  - default_backend: optional backend override
```

### WebUI API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/presets` | List all presets |
| `POST` | `/presets/text` | Create text preset |
| `POST` | `/presets/image` | Upload image preset |
| `DELETE` | `/presets/{type}/{name}` | Delete a preset |
| `GET` | `/presets/image/{name}/preview` | Preview image preset |
| `GET` | `/backends` | List available backends |

## Tools Exposed to the Agent

- `AI 文生图` — Generate an image from a text prompt (uses default backend)
- `AI 单图编辑` — Edit one image with a text prompt
- `AI 多图参考编辑` — Create or edit using multiple reference images
- `AI 预设生图` — Generate using a saved preset

## Running Tests

```bash
cd tests
python -m pytest test_units.py -v
```

## Notes

- The `module_name` remains `gpt_image` for backward compatibility.
- This repository contains only the plugin source code.