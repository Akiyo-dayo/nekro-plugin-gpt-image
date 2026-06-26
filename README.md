# Nekro Agent GPT Image Plugin

A Nekro Agent plugin for generating and editing images with OpenAI-compatible Images API endpoints.

## Features

- Text-to-image generation
- Single-image editing
- Multi-image reference editing
- Configurable model, size, quality, background, output format, moderation, and timeout

## Installation

Copy this plugin directory into your Nekro Agent plugin packages directory:

```bash
plugins/packages/gpt_image
```

Then restart Nekro Agent or reload plugins from the Nekro Agent plugin manager.

## Configuration

Configure the plugin in Nekro Agent after installation.

Required:

- `MODEL_GROUP`: a Nekro Agent model group that provides an OpenAI-compatible `BASE_URL` and `API_KEY`

Optional defaults:

- `MODEL_NAME`: image model name, for example `gpt-image-1.5`
- `DEFAULT_SIZE`: `auto`, `1024x1024`, `1536x1024`, `1024x1536`, etc.
- `DEFAULT_QUALITY`: `auto`, `low`, `medium`, or `high`
- `DEFAULT_BACKGROUND`: `auto`, `transparent`, or `opaque`
- `OUTPUT_FORMAT`: `png`, `jpeg`, or `webp`
- `OUTPUT_COMPRESSION`: JPEG/WebP compression quality from `1` to `100`
- `MODERATION`: `auto` or `low`
- `TIMEOUT_SECONDS`: request timeout in seconds
- `MAX_REFERENCE_IMAGES`: maximum number of reference images for multi-image editing

Example model group values should be configured in Nekro Agent itself. Do not commit real API keys:

```yaml
MODEL_GROUPS:
  image_api:
    BASE_URL: "https://api.example.com/v1"
    API_KEY: "${IMAGE_API_KEY}"
    CHAT_MODEL: "gpt-image-1.5"
```

## Tools exposed to the agent

- `GPT 文生图`: generate an image from a prompt
- `GPT 单图编辑`: edit one image with a prompt
- `GPT 多图参考编辑`: create or edit an image using multiple reference images

## Notes

This repository contains only the plugin source code. Runtime config files, plugin data, caches, logs, and private credentials are intentionally excluded.
