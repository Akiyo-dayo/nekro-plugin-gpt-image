from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional

import base64
import aiofiles
import magic
from httpx import HTTPStatusError, RequestError
from pydantic import Field

from nekro_agent.api import i18n
from nekro_agent.api.plugin import (
    Arg,
    CmdCtl,
    CommandExecutionContext,
    CommandPermission,
    CommandResponse,
    ConfigBase,
    ExtraField,
    NekroPlugin,
    SandboxMethodType,
)
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core import logger
from nekro_agent.core.config import config as global_config
from nekro_agent.services.command.schemas import (
    CommandOutputSegment,
    CommandOutputSegmentType,
)

from .client import (
    OpenAIImageAPIError,
    GeminiImageAPIError,
    SDImageAPIError,
    build_generation_payload,
    build_gemini_generation_payload,
    build_prompt_with_references,
    build_sd_img2img_payload,
    build_sd_txt2img_payload,
    decode_gemini_response,
    decode_image_response,
    decode_sd_response,
    image_file_name,
    make_edit_files,
    normalize_output_format,
    normalize_size,
    post_edit,
    post_gemini_generation,
    post_generation,
    post_sd_img2img,
    post_sd_txt2img,
)
from .presets import ImagePreset, PresetStore, TextPreset


plugin = NekroPlugin(
    name="AI 生图插件",
    module_name="gpt_image",
    description="支持 OpenAI / Gemini / Stable Diffusion 多后端同时配置的图片生成与编辑插件。",
    version="0.3.0",
    author="Akiyo",
    url="https://github.com/Akiyo-dayo/nekro-plugin-gpt-image",
    i18n_name=i18n.i18n_text(zh_CN="AI 生图插件", en_US="AI Image Plugin"),
    i18n_description=i18n.i18n_text(
        zh_CN="支持 OpenAI / Gemini / Stable Diffusion 多后端同时配置的图片生成与编辑插件。",
        en_US="Generate and edit images through OpenAI, Gemini, or Stable Diffusion backends.",
    ),
    allow_sleep=True,
    sleep_brief="提供 AI 文生图、单图编辑和多图参考编辑，支持 /na-gpt /na-gemini /na-sd 直接生图命令。仅在用户明确要求生成或修改图片时激活。",
)

BACKEND_OPENAI = "openai"
BACKEND_GEMINI = "gemini"
BACKEND_SD = "sd_webui"
BACKEND_CHOICES = Literal["openai", "gemini", "sd_webui"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@plugin.mount_config()
class GPTImageConfig(ConfigBase):
    DEFAULT_BACKEND: BACKEND_CHOICES = Field(
        default="openai",
        title="默认画图后端",
        description="AI 自动调用时使用的默认后端。用户也可通过 /na-gpt /na-gemini /na-sd 命令指定后端。",
    )

    # --- OpenAI backend ---
    OPENAI_MODEL_GROUP: str = Field(
        default="",
        title="OpenAI 模型组",
        description="提供 OpenAI 兼容 BASE_URL 和 API_KEY 的模型组。留空则禁用 OpenAI 后端。",
        json_schema_extra=ExtraField(ref_model_groups=True, required=False).model_dump(),
    )
    OPENAI_MODEL_NAME: str = Field(
        default="gpt-image-1",
        title="OpenAI 模型名",
        description="传给 OpenAI Images API 的模型名。",
    )

    # --- Gemini backend ---
    GEMINI_MODEL_GROUP: str = Field(
        default="",
        title="Gemini 模型组",
        description="提供 Gemini BASE_URL 和 API_KEY 的模型组。留空则禁用 Gemini 后端。",
        json_schema_extra=ExtraField(ref_model_groups=True, required=False).model_dump(),
    )
    GEMINI_MODEL_NAME: str = Field(
        default="gemini-2.0-flash-preview-image-generation",
        title="Gemini 模型名",
        description="传给 Gemini API 的模型名。",
    )

    # --- SD WebUI backend ---
    SD_MODEL_GROUP: str = Field(
        default="",
        title="SD WebUI 模型组",
        description="提供 SD WebUI BASE_URL 的模型组（API_KEY 可选）。留空则禁用 SD 后端。",
        json_schema_extra=ExtraField(ref_model_groups=True, required=False).model_dump(),
    )
    SD_MODEL_NAME: str = Field(
        default="",
        title="SD 模型名",
        description="覆盖 SD WebUI 当前加载的模型（checkpoint）。留空则使用 WebUI 当前模型。",
    )
    SD_NEGATIVE_PROMPT: str = Field(
        default="",
        title="SD 负面提示词",
        description="Stable Diffusion 默认负面提示词。",
    )
    SD_STEPS: int = Field(default=20, title="SD 采样步数", ge=1, le=150)
    SD_CFG_SCALE: float = Field(default=7.0, title="SD CFG Scale", ge=1.0, le=30.0)
    SD_SAMPLER: str = Field(default="Euler a", title="SD 采样器")
    SD_DENOISING_STRENGTH: float = Field(
        default=0.75, title="SD 图生图去噪强度", ge=0.0, le=1.0,
    )

    # --- Common settings ---
    DEFAULT_SIZE: str = Field(default="1024x1024", title="默认图片尺寸")
    DEFAULT_QUALITY: Literal["auto", "low", "medium", "high"] = Field(
        default="auto", title="默认质量（OpenAI）",
    )
    DEFAULT_BACKGROUND: Literal["auto", "transparent", "opaque"] = Field(
        default="auto", title="默认背景（OpenAI）",
    )
    OUTPUT_FORMAT: Literal["png", "jpeg", "webp"] = Field(
        default="png", title="输出格式",
    )
    OUTPUT_COMPRESSION: int = Field(default=100, title="JPEG/WebP 压缩质量", ge=1, le=100)
    MODERATION: Literal["auto", "low"] = Field(default="auto", title="审核强度（OpenAI）")
    TIMEOUT_SECONDS: int = Field(default=300, title="请求超时秒数", ge=30, le=900)
    MAX_REFERENCE_IMAGES: int = Field(default=5, title="最大参考图片数", ge=1, le=10)


config: GPTImageConfig = plugin.get_config(GPTImageConfig)

# Lazy-initialized preset store
_preset_store: Optional[PresetStore] = None


def _get_preset_store() -> PresetStore:
    global _preset_store
    if _preset_store is None:
        _preset_store = PresetStore(plugin.get_plugin_data_dir())
    return _preset_store

# ---------------------------------------------------------------------------
# Model group helpers
# ---------------------------------------------------------------------------

def _get_backend_model_group(backend: str):
    """Get model group config for a specific backend."""
    group_key_map = {
        BACKEND_OPENAI: config.OPENAI_MODEL_GROUP,
        BACKEND_GEMINI: config.GEMINI_MODEL_GROUP,
        BACKEND_SD: config.SD_MODEL_GROUP,
    }
    label_map = {BACKEND_OPENAI: "OpenAI", BACKEND_GEMINI: "Gemini", BACKEND_SD: "SD WebUI"}
    label = label_map.get(backend, backend)
    group_key = group_key_map.get(backend, "")
    if not group_key or not group_key.strip():
        raise ValueError(f"{label} 后端未配置模型组，请在插件设置中填写")
    if group_key not in global_config.MODEL_GROUPS:
        raise ValueError(f"模型组 `{group_key}` 未在 Nekro Agent 中配置")
    mg = global_config.MODEL_GROUPS[group_key]
    if backend != BACKEND_SD and not getattr(mg, "API_KEY", ""):
        raise ValueError(f"模型组 `{group_key}` 缺少 API_KEY")
    if not getattr(mg, "BASE_URL", ""):
        raise ValueError(f"模型组 `{group_key}` 缺少 BASE_URL")
    return mg


def _get_model_name(backend: str, model_group) -> str:
    name_map = {
        BACKEND_OPENAI: config.OPENAI_MODEL_NAME,
        BACKEND_GEMINI: config.GEMINI_MODEL_NAME,
        BACKEND_SD: config.SD_MODEL_NAME,
    }
    name = (name_map.get(backend, "") or getattr(model_group, "CHAT_MODEL", "")).strip()
    if not name and backend == BACKEND_OPENAI:
        name = "gpt-image-1"
    elif not name and backend == BACKEND_GEMINI:
        name = "gemini-2.0-flash-preview-image-generation"
    return name


def _friendly_api_error(exc: Exception) -> str:
    if isinstance(exc, HTTPStatusError):
        status = exc.response.status_code
        try:
            body = exc.response.json()
        except Exception:
            body = exc.response.text[:500]
        return f"Image API HTTP {status}: {body}"
    if isinstance(exc, RequestError):
        return f"Image API request failed: {exc}"
    return str(exc)


_BACKEND_LABEL = {"openai": "GPT", "gemini": "Gemini", "sd_webui": "SD"}


def _label(backend: str = "") -> str:
    return _BACKEND_LABEL.get(backend or config.DEFAULT_BACKEND, "AI")


def _available_backends() -> List[str]:
    backends = []
    if config.OPENAI_MODEL_GROUP.strip():
        backends.append(BACKEND_OPENAI)
    if config.GEMINI_MODEL_GROUP.strip():
        backends.append(BACKEND_GEMINI)
    if config.SD_MODEL_GROUP.strip():
        backends.append(BACKEND_SD)
    return backends


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

async def _reference_image_from_path(
    _ctx: AgentCtx, image_path: str, *, name_hint: str,
) -> tuple[str, str, bytes]:
    host_path = Path(_ctx.fs.get_file(image_path))
    async with aiofiles.open(host_path, "rb") as file:
        image_bytes = await file.read()
    mime_type = magic.from_buffer(image_bytes, mime=True) or "application/octet-stream"
    if not mime_type.startswith("image/"):
        raise ValueError(f"{image_path} 不是可用图片文件")
    return mime_type, name_hint or host_path.name, image_bytes


async def _forward_result(_ctx: AgentCtx, image_data: str, output_format: str) -> str:
    return await _ctx.fs.mixed_forward_file(
        image_data, file_name=image_file_name(output_format),
    )


# ---------------------------------------------------------------------------
# Backend dispatch - generation
# ---------------------------------------------------------------------------

async def _generate_openai(*, prompt: str, size: str, quality: str, background: str,
                           output_format: str, output_compression: int, moderation: str) -> str:
    mg = _get_backend_model_group(BACKEND_OPENAI)
    payload = build_generation_payload(
        model=_get_model_name(BACKEND_OPENAI, mg), prompt=prompt, size=size,
        quality=quality, background=background, output_format=output_format,
        output_compression=output_compression, moderation=moderation,
    )
    data = await post_generation(
        base_url=mg.BASE_URL, api_key=mg.API_KEY,
        payload=payload, timeout_seconds=config.TIMEOUT_SECONDS,
    )
    return decode_image_response(data, output_format=output_format)


async def _generate_gemini(*, prompt: str, size: str, quality: str, output_format: str) -> str:
    mg = _get_backend_model_group(BACKEND_GEMINI)
    payload = build_gemini_generation_payload(
        prompt=prompt, size=size, quality=quality, output_format=output_format,
    )
    data = await post_gemini_generation(
        base_url=mg.BASE_URL, api_key=mg.API_KEY,
        model=_get_model_name(BACKEND_GEMINI, mg),
        payload=payload, timeout_seconds=config.TIMEOUT_SECONDS,
    )
    return decode_gemini_response(data, output_format=output_format)


async def _generate_sd(*, prompt: str, negative_prompt: str, size: str, output_format: str) -> str:
    mg = _get_backend_model_group(BACKEND_SD)
    payload = build_sd_txt2img_payload(
        prompt=prompt, negative_prompt=negative_prompt or config.SD_NEGATIVE_PROMPT,
        size=size, steps=config.SD_STEPS, cfg_scale=config.SD_CFG_SCALE,
        sampler_name=config.SD_SAMPLER, sd_model=_get_model_name(BACKEND_SD, mg),
    )
    data = await post_sd_txt2img(
        base_url=mg.BASE_URL, api_key=getattr(mg, "API_KEY", ""),
        payload=payload, timeout_seconds=config.TIMEOUT_SECONDS,
    )
    return decode_sd_response(data, output_format=output_format)


async def _generate_raw_image(
    *, backend: str, prompt: str, size: str, quality: str, background: str,
    output_format: str, output_compression: int, moderation: str,
    negative_prompt: str = "",
) -> str:
    if backend == BACKEND_OPENAI:
        return await _generate_openai(
            prompt=prompt, size=size, quality=quality, background=background,
            output_format=output_format, output_compression=output_compression,
            moderation=moderation,
        )
    elif backend == BACKEND_GEMINI:
        return await _generate_gemini(
            prompt=prompt, size=size, quality=quality, output_format=output_format,
        )
    elif backend == BACKEND_SD:
        return await _generate_sd(
            prompt=prompt, negative_prompt=negative_prompt,
            size=size, output_format=output_format,
        )
    raise ValueError(f"不支持的后端: {backend}")

# ---------------------------------------------------------------------------
# Backend dispatch - edit (with reference images)
# ---------------------------------------------------------------------------

async def _edit_openai(
    *, _ctx: AgentCtx, prompt: str, reference_images: List[Mapping[str, Any]],
    size: str, quality: str, background: str,
    output_format: str, output_compression: int, moderation: str,
) -> str:
    mg = _get_backend_model_group(BACKEND_OPENAI)
    normalized_size = normalize_size(size)
    fmt = normalize_output_format(output_format)
    image_items = []
    for index, item in enumerate(reference_images, 1):
        image_path = str(item.get("image_path") or "").strip()
        if not image_path:
            raise ValueError(f"第 {index} 张参考图缺少 image_path")
        image_items.append(
            await _reference_image_from_path(_ctx, image_path, name_hint=f"reference_{index}"),
        )
    fields: Dict[str, Any] = {
        "model": _get_model_name(BACKEND_OPENAI, mg),
        "prompt": prompt, "n": 1, "output_format": fmt,
    }
    if normalized_size != "auto":
        fields["size"] = normalized_size
    if quality != "auto":
        fields["quality"] = quality
    if background != "auto":
        fields["background"] = background
    if moderation != "auto":
        fields["moderation"] = moderation
    if fmt in {"jpeg", "webp"} and output_compression < 100:
        fields["output_compression"] = output_compression
    data = await post_edit(
        base_url=mg.BASE_URL, api_key=mg.API_KEY,
        fields=fields, files=make_edit_files(image_items),
        timeout_seconds=config.TIMEOUT_SECONDS,
    )
    return decode_image_response(data, output_format=fmt)


async def _edit_gemini(
    *, _ctx: AgentCtx, prompt: str, reference_images: List[Mapping[str, Any]],
    size: str, quality: str, output_format: str,
) -> str:
    mg = _get_backend_model_group(BACKEND_GEMINI)
    ref_data: List[tuple[str, bytes]] = []
    for index, item in enumerate(reference_images, 1):
        image_path = str(item.get("image_path") or "").strip()
        if not image_path:
            raise ValueError(f"第 {index} 张参考图缺少 image_path")
        mime_type, _, img_bytes = await _reference_image_from_path(
            _ctx, image_path, name_hint=f"reference_{index}",
        )
        ref_data.append((mime_type, img_bytes))
    payload = build_gemini_generation_payload(
        prompt=prompt, size=size, quality=quality,
        output_format=output_format, reference_images=ref_data,
    )
    data = await post_gemini_generation(
        base_url=mg.BASE_URL, api_key=mg.API_KEY,
        model=_get_model_name(BACKEND_GEMINI, mg),
        payload=payload, timeout_seconds=config.TIMEOUT_SECONDS,
    )
    return decode_gemini_response(data, output_format=output_format)


async def _edit_sd(
    *, _ctx: AgentCtx, prompt: str, reference_images: List[Mapping[str, Any]],
    size: str, output_format: str, negative_prompt: str = "",
) -> str:
    mg = _get_backend_model_group(BACKEND_SD)
    init_images_b64: List[str] = []
    for index, item in enumerate(reference_images, 1):
        image_path = str(item.get("image_path") or "").strip()
        if not image_path:
            raise ValueError(f"第 {index} 张参考图缺少 image_path")
        _, _, img_bytes = await _reference_image_from_path(
            _ctx, image_path, name_hint=f"reference_{index}",
        )
        init_images_b64.append(base64.b64encode(img_bytes).decode())
    payload = build_sd_img2img_payload(
        prompt=prompt, negative_prompt=negative_prompt or config.SD_NEGATIVE_PROMPT,
        init_images_b64=init_images_b64, size=size, steps=config.SD_STEPS,
        cfg_scale=config.SD_CFG_SCALE, denoising_strength=config.SD_DENOISING_STRENGTH,
        sampler_name=config.SD_SAMPLER, sd_model=_get_model_name(BACKEND_SD, mg),
    )
    data = await post_sd_img2img(
        base_url=mg.BASE_URL, api_key=getattr(mg, "API_KEY", ""),
        payload=payload, timeout_seconds=config.TIMEOUT_SECONDS,
    )
    return decode_sd_response(data, output_format=output_format)


async def _edit_raw_image(
    *, _ctx: AgentCtx, backend: str, prompt: str,
    reference_images: List[Mapping[str, Any]],
    size: str, quality: str, background: str,
    output_format: str, output_compression: int, moderation: str,
    negative_prompt: str = "",
) -> str:
    if not reference_images:
        raise ValueError("至少需要一张参考图片")
    if len(reference_images) > config.MAX_REFERENCE_IMAGES:
        raise ValueError(f"最多支持 {config.MAX_REFERENCE_IMAGES} 张参考图片")
    if backend == BACKEND_OPENAI:
        return await _edit_openai(
            _ctx=_ctx, prompt=prompt, reference_images=reference_images,
            size=size, quality=quality, background=background,
            output_format=output_format, output_compression=output_compression,
            moderation=moderation,
        )
    elif backend == BACKEND_GEMINI:
        return await _edit_gemini(
            _ctx=_ctx, prompt=prompt, reference_images=reference_images,
            size=size, quality=quality, output_format=output_format,
        )
    elif backend == BACKEND_SD:
        return await _edit_sd(
            _ctx=_ctx, prompt=prompt, reference_images=reference_images,
            size=size, output_format=output_format, negative_prompt=negative_prompt,
        )
    raise ValueError(f"不支持的后端: {backend}")

# ---------------------------------------------------------------------------
# Sandbox methods (AI auto-call)
# ---------------------------------------------------------------------------

@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL, name="AI 文生图",
    description="使用 AI 图像模型生成图片（支持 GPT / Gemini / SD，按插件默认后端）",
)
async def ai_image_generate(
    _ctx: AgentCtx,
    prompt: str,
    size: str = "",
    quality: Literal["auto", "low", "medium", "high"] = "auto",
    background: Literal["auto", "transparent", "opaque"] = "auto",
    send_to_chat: bool = True,
) -> str:
    """Generate an image with an AI image model.

    Args:
        prompt: Detailed image prompt.
        size: Optional size, "auto" or WIDTHxHEIGHT. Empty uses plugin default.
        quality: auto / low / medium / high (OpenAI only).
        background: auto / transparent / opaque (OpenAI only).
        send_to_chat: Send the generated image to chat.

    Returns:
        The generated image sandbox path.
    """
    backend = config.DEFAULT_BACKEND
    try:
        image_data = await _generate_raw_image(
            backend=backend, prompt=prompt,
            size=size or config.DEFAULT_SIZE,
            quality=quality if quality != "auto" else config.DEFAULT_QUALITY,
            background=background if background != "auto" else config.DEFAULT_BACKGROUND,
            output_format=config.OUTPUT_FORMAT,
            output_compression=config.OUTPUT_COMPRESSION,
            moderation=config.MODERATION,
        )
        path = await _forward_result(_ctx, image_data, config.OUTPUT_FORMAT)
        if send_to_chat:
            await _ctx.send_image(path)
        return path
    except (OpenAIImageAPIError, GeminiImageAPIError, SDImageAPIError,
            HTTPStatusError, RequestError, ValueError) as exc:
        logger.error(f"{_label(backend)} 文生图失败: {_friendly_api_error(exc)}")
        raise Exception(f"{_label(backend)} 文生图失败: {_friendly_api_error(exc)}") from exc


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL, name="AI 单图编辑",
    description="使用 AI 图像模型编辑一张图片（支持 GPT / Gemini / SD）",
)
async def ai_image_edit(
    _ctx: AgentCtx,
    image_path: str,
    prompt: str,
    size: str = "",
    quality: Literal["auto", "low", "medium", "high"] = "auto",
    background: Literal["auto", "transparent", "opaque"] = "auto",
    send_to_chat: bool = True,
) -> str:
    """Edit one reference image with an AI image model.

    Args:
        image_path: Sandbox image path.
        prompt: Editing instruction.
        size: Optional size.
        quality: auto / low / medium / high (OpenAI only).
        background: auto / transparent / opaque (OpenAI only).
        send_to_chat: Send the edited image to chat.

    Returns:
        The edited image sandbox path.
    """
    return await ai_image_multi_edit(
        _ctx,
        reference_images=[{"image_path": image_path, "description": "source image", "weight": 1.0}],
        target_prompt=prompt, size=size, quality=quality,
        background=background, send_to_chat=send_to_chat,
    )


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL, name="AI 多图参考编辑",
    description="使用多张参考图片生成或编辑图片（支持 GPT / Gemini / SD）",
)
async def ai_image_multi_edit(
    _ctx: AgentCtx,
    reference_images: List[Dict[str, Any]],
    target_prompt: str,
    size: str = "",
    quality: Literal["auto", "low", "medium", "high"] = "auto",
    background: Literal["auto", "transparent", "opaque"] = "auto",
    send_to_chat: bool = True,
) -> str:
    """Create or edit an image using multiple reference images.

    Args:
        reference_images: List of dicts with image_path, description, weight.
        target_prompt: Desired final image description.
        size: Optional size.
        quality: auto / low / medium / high (OpenAI only).
        background: auto / transparent / opaque (OpenAI only).
        send_to_chat: Send the result to chat.

    Returns:
        The generated or edited image sandbox path.
    """
    backend = config.DEFAULT_BACKEND
    try:
        final_size = size or config.DEFAULT_SIZE
        prompt = build_prompt_with_references(
            target_prompt=target_prompt, references=reference_images, size=final_size,
        )
        image_data = await _edit_raw_image(
            _ctx=_ctx, backend=backend, prompt=prompt,
            reference_images=reference_images, size=final_size,
            quality=quality if quality != "auto" else config.DEFAULT_QUALITY,
            background=background if background != "auto" else config.DEFAULT_BACKGROUND,
            output_format=config.OUTPUT_FORMAT,
            output_compression=config.OUTPUT_COMPRESSION,
            moderation=config.MODERATION,
        )
        path = await _forward_result(_ctx, image_data, config.OUTPUT_FORMAT)
        if send_to_chat:
            await _ctx.send_image(path)
        return path
    except (OpenAIImageAPIError, GeminiImageAPIError, SDImageAPIError,
            HTTPStatusError, RequestError, ValueError) as exc:
        logger.error(f"{_label(backend)} 图像编辑失败: {_friendly_api_error(exc)}")
        raise Exception(f"{_label(backend)} 图像编辑失败: {_friendly_api_error(exc)}") from exc


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL, name="AI 预设生图",
    description="使用已保存的预设快速生成图片（支持图片预设和文本预设）",
)
async def ai_preset_generate(
    _ctx: AgentCtx,
    preset_name: str,
    user_input: str = "",
    send_to_chat: bool = True,
) -> str:
    """Generate image using a saved preset.

    Args:
        preset_name: Name of the preset to use.
        user_input: Extra user input; for text presets replaces {input} in template.
        send_to_chat: Send the result to chat.

    Returns:
        The generated image sandbox path.
    """
    store = _get_preset_store()

    text_preset = store.get_text_preset(preset_name)
    if text_preset:
        prompt = text_preset.prompt_template.replace("{input}", user_input) if user_input else text_preset.prompt_template.replace("{input}", "")
        backend = text_preset.default_backend or config.DEFAULT_BACKEND
        size = text_preset.default_size or config.DEFAULT_SIZE
        image_data = await _generate_raw_image(
            backend=backend, prompt=prompt,
            size=size, quality=text_preset.default_quality or config.DEFAULT_QUALITY,
            background=config.DEFAULT_BACKGROUND,
            output_format=config.OUTPUT_FORMAT,
            output_compression=config.OUTPUT_COMPRESSION,
            moderation=config.MODERATION,
            negative_prompt=text_preset.negative_prompt,
        )
        path = await _forward_result(_ctx, image_data, config.OUTPUT_FORMAT)
        if send_to_chat:
            await _ctx.send_image(path)
        return path

    image_preset = store.get_image_preset(preset_name)
    if image_preset:
        prompt = user_input or image_preset.default_prompt or "根据参考图片生成类似风格的图片"
        backend = image_preset.default_backend or config.DEFAULT_BACKEND
        size = image_preset.default_size or config.DEFAULT_SIZE
        data_uri = f"data:{image_preset.mime_type};base64,{image_preset.image_b64}"
        ref_path = await _ctx.fs.mixed_forward_file(data_uri, file_name="preset_ref.png")
        return await ai_image_edit(
            _ctx, image_path=ref_path, prompt=prompt,
            size=size, send_to_chat=send_to_chat,
        )

    raise ValueError(f"预设 '{preset_name}' 不存在，请先创建预设")

# ---------------------------------------------------------------------------
# Slash commands: /na-gpt, /na-gemini, /na-sd
# ---------------------------------------------------------------------------

async def _cmd_generate(context: CommandExecutionContext, backend: str, prompt: str) -> CommandResponse:
    """Shared logic for /na-gpt, /na-gemini, /na-sd commands."""
    if not prompt.strip():
        return CmdCtl.failed(f"请提供生图提示词，例如: /na-{_label(backend).lower()} 一只可爱的猫咪")

    try:
        _get_backend_model_group(backend)
    except ValueError as exc:
        return CmdCtl.failed(str(exc))

    store = _get_preset_store()
    text_preset = store.get_text_preset(prompt.strip())
    image_preset = store.get_image_preset(prompt.strip())

    actual_prompt = prompt.strip()
    negative_prompt = ""
    size = config.DEFAULT_SIZE

    if text_preset:
        actual_prompt = text_preset.prompt_template.replace("{input}", "")
        negative_prompt = text_preset.negative_prompt
        size = text_preset.default_size or size
    elif image_preset:
        actual_prompt = image_preset.default_prompt or "根据参考图片生成类似风格的图片"
        size = image_preset.default_size or size

    try:
        if image_preset and not text_preset:
            data_uri = f"data:{image_preset.mime_type};base64,{image_preset.image_b64}"
            fmt = normalize_output_format(config.OUTPUT_FORMAT)
            image_data = await _generate_raw_image(
                backend=backend, prompt=actual_prompt,
                size=size, quality=config.DEFAULT_QUALITY,
                background=config.DEFAULT_BACKGROUND,
                output_format=config.OUTPUT_FORMAT,
                output_compression=config.OUTPUT_COMPRESSION,
                moderation=config.MODERATION,
                negative_prompt=negative_prompt,
            )
        else:
            image_data = await _generate_raw_image(
                backend=backend, prompt=actual_prompt,
                size=size, quality=config.DEFAULT_QUALITY,
                background=config.DEFAULT_BACKGROUND,
                output_format=config.OUTPUT_FORMAT,
                output_compression=config.OUTPUT_COMPRESSION,
                moderation=config.MODERATION,
                negative_prompt=negative_prompt,
            )
    except (OpenAIImageAPIError, GeminiImageAPIError, SDImageAPIError,
            HTTPStatusError, RequestError, ValueError) as exc:
        return CmdCtl.failed(f"{_label(backend)} 生图失败: {_friendly_api_error(exc)}")

    from .client import decode_data_uri
    try:
        mime_type, img_bytes = decode_data_uri(image_data)
        ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(mime_type, "png")
        save_dir = plugin.get_plugin_data_dir() / "generated"
        save_dir.mkdir(parents=True, exist_ok=True)
        import time as _time
        file_name = f"{_label(backend).lower()}_{int(_time.time())}.{ext}"
        file_path = save_dir / file_name
        file_path.write_bytes(img_bytes)
        abs_path = str(file_path.resolve())

        return CmdCtl.success([
            CommandOutputSegment(
                type=CommandOutputSegmentType.TEXT,
                text=f"{_label(backend)} 生图完成",
            ),
            CommandOutputSegment(
                type=CommandOutputSegmentType.IMAGE,
                file_path=abs_path,
            ),
        ])
    except Exception:
        return CmdCtl.success(f"{_label(backend)} 生图完成（图片为 URL）: {image_data[:200]}")


@plugin.mount_command(
    name="na-gpt",
    description="使用 GPT (OpenAI) 后端生成图片",
    aliases=["gpt画图", "gpt生图"],
    permission=CommandPermission.PUBLIC,
    usage="na-gpt <提示词>",
)
async def cmd_na_gpt(
    context: CommandExecutionContext,
    prompt: Annotated[str, Arg("生图提示词", positional=True, greedy=True)] = "",
) -> CommandResponse:
    return await _cmd_generate(context, BACKEND_OPENAI, prompt)


@plugin.mount_command(
    name="na-gemini",
    description="使用 Gemini 后端生成图片",
    aliases=["gemini画图", "gemini生图"],
    permission=CommandPermission.PUBLIC,
    usage="na-gemini <提示词>",
)
async def cmd_na_gemini(
    context: CommandExecutionContext,
    prompt: Annotated[str, Arg("生图提示词", positional=True, greedy=True)] = "",
) -> CommandResponse:
    return await _cmd_generate(context, BACKEND_GEMINI, prompt)


@plugin.mount_command(
    name="na-sd",
    description="使用 Stable Diffusion 后端生成图片",
    aliases=["sd画图", "sd生图"],
    permission=CommandPermission.PUBLIC,
    usage="na-sd <提示词>",
)
async def cmd_na_sd(
    context: CommandExecutionContext,
    prompt: Annotated[str, Arg("生图提示词", positional=True, greedy=True)] = "",
) -> CommandResponse:
    return await _cmd_generate(context, BACKEND_SD, prompt)


# ---------------------------------------------------------------------------
# Preset management commands
# ---------------------------------------------------------------------------

preset_group = plugin.mount_command_group(
    name="na-preset",
    description="管理生图预设（图片预设和文本预设）",
    permission=CommandPermission.ADVANCED,
)


@preset_group.command(name="list", description="列出所有预设")
async def cmd_preset_list(context: CommandExecutionContext) -> CommandResponse:
    store = _get_preset_store()
    image_presets = store.list_image_presets()
    text_presets = store.list_text_presets()

    lines = []
    if text_presets:
        lines.append("📝 文本预设:")
        for p in text_presets:
            desc = f" - {p['description']}" if p.get("description") else ""
            backend = f" [{p['default_backend']}]" if p.get("default_backend") else ""
            lines.append(f"  · {p['name']}{desc}{backend}")
    if image_presets:
        lines.append("🖼️ 图片预设:")
        for p in image_presets:
            desc = f" - {p['description']}" if p.get("description") else ""
            backend = f" [{p['default_backend']}]" if p.get("default_backend") else ""
            lines.append(f"  · {p['name']}{desc}{backend}")
    if not lines:
        return CmdCtl.success("暂无预设。使用 /na-preset.add-text 或通过 WebUI 上传图片预设。")
    return CmdCtl.success("\n".join(lines))


@preset_group.command(name="add-text", description="添加文本预设")
async def cmd_preset_add_text(
    context: CommandExecutionContext,
    name: Annotated[str, Arg("预设名称", positional=True)] = "",
    prompt: Annotated[str, Arg("提示词模板，用 {input} 表示用户输入占位", positional=True, greedy=True)] = "",
) -> CommandResponse:
    if not name.strip() or not prompt.strip():
        return CmdCtl.failed("用法: /na-preset.add-text <名称> <提示词模板>\n示例: /na-preset.add-text 动漫风 将以下内容画成动漫风格: {input}")
    store = _get_preset_store()
    preset = TextPreset(name=name.strip(), prompt_template=prompt.strip())
    store.save_text_preset(preset)
    return CmdCtl.success(f"文本预设 '{name.strip()}' 已保存")


@preset_group.command(name="delete", description="删除预设")
async def cmd_preset_delete(
    context: CommandExecutionContext,
    name: Annotated[str, Arg("预设名称", positional=True)] = "",
) -> CommandResponse:
    if not name.strip():
        return CmdCtl.failed("请指定预设名称")
    store = _get_preset_store()
    deleted = store.delete_text_preset(name.strip()) or store.delete_image_preset(name.strip())
    if deleted:
        return CmdCtl.success(f"预设 '{name.strip()}' 已删除")
    return CmdCtl.failed(f"预设 '{name.strip()}' 不存在")

# ---------------------------------------------------------------------------
# WebUI routes (preset upload / management via HTTP API)
# ---------------------------------------------------------------------------

@plugin.mount_router()
def create_router():
    from fastapi import APIRouter, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    router = APIRouter()

    # -- WebUI 管理页面 --
    @router.get("/manage", response_class=HTMLResponse)
    async def manage_page():
        html_path = Path(__file__).parent / "webui.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    # -- JSON API --

    @router.get("/presets")
    async def list_presets():
        store = _get_preset_store()
        return JSONResponse({
            "image_presets": store.list_image_presets(),
            "text_presets": store.list_text_presets(),
            "available_backends": _available_backends(),
        })

    @router.post("/presets/text")
    async def create_text_preset(
        name: str = Form(...),
        prompt_template: str = Form(...),
        description: str = Form(""),
        negative_prompt: str = Form(""),
        default_size: str = Form(""),
        default_quality: str = Form("auto"),
        default_backend: str = Form(""),
    ):
        store = _get_preset_store()
        preset = TextPreset(
            name=name.strip(),
            description=description.strip(),
            prompt_template=prompt_template.strip(),
            negative_prompt=negative_prompt.strip(),
            default_size=default_size.strip(),
            default_quality=default_quality.strip() or "auto",
            default_backend=default_backend.strip(),
        )
        store.save_text_preset(preset)
        return JSONResponse({"status": "ok", "name": preset.name})

    @router.post("/presets/image")
    async def create_image_preset(
        name: str = Form(...),
        file: UploadFile = File(...),
        description: str = Form(""),
        default_prompt: str = Form(""),
        default_size: str = Form(""),
        default_backend: str = Form(""),
    ):
        import base64 as b64mod
        content = await file.read()
        mime_type = file.content_type or "image/png"
        if not mime_type.startswith("image/"):
            return JSONResponse({"status": "error", "message": "上传文件不是图片"}, status_code=400)
        image_b64 = b64mod.b64encode(content).decode()
        store = _get_preset_store()
        preset = ImagePreset(
            name=name.strip(),
            description=description.strip(),
            image_b64=image_b64,
            mime_type=mime_type,
            default_prompt=default_prompt.strip(),
            default_size=default_size.strip(),
            default_backend=default_backend.strip(),
        )
        store.save_image_preset(preset)
        return JSONResponse({"status": "ok", "name": preset.name})

    @router.delete("/presets/{preset_type}/{name}")
    async def delete_preset(preset_type: str, name: str):
        store = _get_preset_store()
        if preset_type == "text":
            ok = store.delete_text_preset(name)
        elif preset_type == "image":
            ok = store.delete_image_preset(name)
        else:
            return JSONResponse({"status": "error", "message": "类型须为 text 或 image"}, status_code=400)
        if ok:
            return JSONResponse({"status": "ok"})
        return JSONResponse({"status": "error", "message": f"预设 '{name}' 不存在"}, status_code=404)

    @router.get("/presets/image/{name}/preview")
    async def preview_image_preset(name: str):
        store = _get_preset_store()
        preset = store.get_image_preset(name)
        if not preset:
            return JSONResponse({"status": "error", "message": "预设不存在"}, status_code=404)
        import base64 as b64mod
        img_bytes = b64mod.b64decode(preset.image_b64)
        return Response(content=img_bytes, media_type=preset.mime_type)

    @router.get("/backends")
    async def list_backends():
        return JSONResponse({
            "available": _available_backends(),
            "default": config.DEFAULT_BACKEND,
        })

    return router


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@plugin.mount_collect_methods()
async def collect_available_methods(_ctx: AgentCtx) -> List[Any]:
    return [ai_image_generate, ai_image_edit, ai_image_multi_edit, ai_preset_generate]


@plugin.mount_cleanup_method()
async def clean_up():
    """Clean up plugin resources."""