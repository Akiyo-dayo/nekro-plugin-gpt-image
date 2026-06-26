from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping

import aiofiles
import magic
from httpx import HTTPStatusError, RequestError
from pydantic import Field

from nekro_agent.api import i18n
from nekro_agent.api.plugin import ConfigBase, ExtraField, NekroPlugin, SandboxMethodType
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core import logger
from nekro_agent.core.config import config as global_config

from .client import (
    OpenAIImageAPIError,
    build_generation_payload,
    build_prompt_with_references,
    decode_image_response,
    image_file_name,
    make_edit_files,
    normalize_output_format,
    normalize_size,
    post_edit,
    post_generation,
)


plugin = NekroPlugin(
    name="GPT 生图插件",
    module_name="gpt_image",
    description="使用 OpenAI Images API 生成、编辑和多图参考生成图片。",
    version="0.1.0",
    author="Akiyo",
    url="https://github.com/Akiyo-dayo/nekro-plugin-gpt-image",
    i18n_name=i18n.i18n_text(zh_CN="GPT 生图插件", en_US="GPT Image Plugin"),
    i18n_description=i18n.i18n_text(
        zh_CN="使用 OpenAI Images API 生成、编辑和多图参考生成图片。",
        en_US="Generate and edit images through the OpenAI Images API.",
    ),
    allow_sleep=True,
    sleep_brief="提供 GPT 文生图、单图编辑和多图参考编辑。仅在用户明确要求生成或修改图片时激活。",
)


@plugin.mount_config()
class GPTImageConfig(ConfigBase):
    MODEL_GROUP: str = Field(
        default="",
        title="接口模型组",
        description="提供 OpenAI 兼容 BASE_URL 和 API_KEY 的模型组。留空时插件不会调用任何模型组。",
        json_schema_extra=ExtraField(ref_model_groups=True, required=False).model_dump(),
    )
    MODEL_NAME: str = Field(
        default="gpt-image-1.5",
        title="图像模型名",
        description="传给 Images API 的模型名，例如 gpt-image-1.5。留空时使用模型组的 CHAT_MODEL。",
    )
    DEFAULT_SIZE: str = Field(
        default="1024x1024",
        title="默认图片尺寸",
        description='支持 auto 或 WIDTHxHEIGHT，例如 "1024x1024"、"1536x1024"、"1024x1536"。',
    )
    DEFAULT_QUALITY: Literal["auto", "low", "medium", "high"] = Field(
        default="auto",
        title="默认质量",
        description="传给 Images API 的 quality 参数。",
    )
    DEFAULT_BACKGROUND: Literal["auto", "transparent", "opaque"] = Field(
        default="auto",
        title="默认背景",
        description="transparent 适合 PNG/WebP 透明背景需求。",
    )
    OUTPUT_FORMAT: Literal["png", "jpeg", "webp"] = Field(
        default="png",
        title="输出格式",
        description="生成图片保存格式。",
    )
    OUTPUT_COMPRESSION: int = Field(
        default=100,
        title="JPEG/WebP 压缩质量",
        description="仅 output_format 为 jpeg 或 webp 且小于 100 时发送。",
        ge=1,
        le=100,
    )
    MODERATION: Literal["auto", "low"] = Field(
        default="auto",
        title="审核强度",
        description="传给 Images API 的 moderation 参数。",
    )
    TIMEOUT_SECONDS: int = Field(
        default=300,
        title="请求超时秒数",
        description="生成或编辑图片的 HTTP 超时时间。",
        ge=30,
        le=900,
    )
    MAX_REFERENCE_IMAGES: int = Field(
        default=5,
        title="最大参考图片数",
        description="多图编辑允许的最大参考图片数量。",
        ge=1,
        le=10,
    )


config: GPTImageConfig = plugin.get_config(GPTImageConfig)


def _get_model_group():
    if not config.MODEL_GROUP.strip():
        raise ValueError("请先在插件配置中选择模型组")
    if config.MODEL_GROUP not in global_config.MODEL_GROUPS:
        raise ValueError(f"模型组 `{config.MODEL_GROUP}` 未配置")
    model_group = global_config.MODEL_GROUPS[config.MODEL_GROUP]
    if not getattr(model_group, "API_KEY", ""):
        raise ValueError(f"模型组 `{config.MODEL_GROUP}` 缺少 API_KEY")
    if not getattr(model_group, "BASE_URL", ""):
        raise ValueError(f"模型组 `{config.MODEL_GROUP}` 缺少 BASE_URL")
    return model_group


def _model_name(model_group) -> str:
    return (config.MODEL_NAME or getattr(model_group, "CHAT_MODEL", "")).strip()


def _friendly_api_error(exc: Exception) -> str:
    if isinstance(exc, HTTPStatusError):
        status = exc.response.status_code
        try:
            body = exc.response.json()
        except Exception:
            body = exc.response.text[:500]
        return f"GPT image API returned HTTP {status}: {body}"
    if isinstance(exc, RequestError):
        return f"GPT image API request failed: {exc}"
    return str(exc)


async def _reference_image_from_path(_ctx: AgentCtx, image_path: str, *, name_hint: str) -> tuple[str, str, bytes]:
    host_path = Path(_ctx.fs.get_file(image_path))
    async with aiofiles.open(host_path, "rb") as file:
        image_bytes = await file.read()
    mime_type = magic.from_buffer(image_bytes, mime=True) or "application/octet-stream"
    if not mime_type.startswith("image/"):
        raise ValueError(f"{image_path} 不是可用图片文件")
    return mime_type, name_hint or host_path.name, image_bytes


async def _forward_result(_ctx: AgentCtx, image_data: str, output_format: str) -> str:
    return await _ctx.fs.mixed_forward_file(image_data, file_name=image_file_name(output_format))


async def _generate_raw_image(
    *,
    prompt: str,
    size: str,
    quality: str,
    background: str,
    output_format: str,
    output_compression: int,
    moderation: str,
) -> str:
    model_group = _get_model_group()
    payload = build_generation_payload(
        model=_model_name(model_group),
        prompt=prompt,
        size=size,
        quality=quality,
        background=background,
        output_format=output_format,
        output_compression=output_compression,
        moderation=moderation,
    )
    data = await post_generation(
        base_url=model_group.BASE_URL,
        api_key=model_group.API_KEY,
        payload=payload,
        timeout_seconds=config.TIMEOUT_SECONDS,
    )
    return decode_image_response(data, output_format=output_format)


async def _edit_raw_image(
    *,
    _ctx: AgentCtx,
    prompt: str,
    reference_images: List[Mapping[str, Any]],
    size: str,
    quality: str,
    background: str,
    output_format: str,
    output_compression: int,
    moderation: str,
) -> str:
    if not reference_images:
        raise ValueError("至少需要一张参考图片")
    if len(reference_images) > config.MAX_REFERENCE_IMAGES:
        raise ValueError(f"最多支持 {config.MAX_REFERENCE_IMAGES} 张参考图片")

    model_group = _get_model_group()
    normalized_size = normalize_size(size)
    fmt = normalize_output_format(output_format)
    image_items = []
    for index, item in enumerate(reference_images, 1):
        image_path = str(item.get("image_path") or "").strip()
        if not image_path:
            raise ValueError(f"第 {index} 张参考图缺少 image_path")
        image_items.append(
            await _reference_image_from_path(
                _ctx,
                image_path,
                name_hint=f"reference_{index}",
            ),
        )

    fields: Dict[str, Any] = {
        "model": _model_name(model_group),
        "prompt": prompt,
        "n": 1,
        "output_format": fmt,
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
        base_url=model_group.BASE_URL,
        api_key=model_group.API_KEY,
        fields=fields,
        files=make_edit_files(image_items),
        timeout_seconds=config.TIMEOUT_SECONDS,
    )
    return decode_image_response(data, output_format=fmt)


@plugin.mount_sandbox_method(SandboxMethodType.TOOL, name="GPT 文生图", description="使用 GPT 图像模型生成图片")
async def gpt_image_generate(
    _ctx: AgentCtx,
    prompt: str,
    size: str = "",
    quality: Literal["auto", "low", "medium", "high"] = "auto",
    background: Literal["auto", "transparent", "opaque"] = "auto",
    send_to_chat: bool = True,
) -> str:
    """Generate an image with a GPT image model.

    Args:
        prompt: Detailed image prompt. Include subject, scene, style, camera/framing, and important constraints.
        size: Optional size. Use "auto" or WIDTHxHEIGHT. Empty uses plugin default.
        quality: Optional quality: auto, low, medium, high.
        background: Optional background: auto, transparent, opaque.
        send_to_chat: Send the generated image to the current chat after generation.

    Returns:
        The generated image sandbox path.
    """
    try:
        image_data = await _generate_raw_image(
            prompt=prompt,
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
    except (OpenAIImageAPIError, HTTPStatusError, RequestError, ValueError) as exc:
        logger.error(f"GPT 文生图失败: {_friendly_api_error(exc)}")
        raise Exception(f"GPT 文生图失败: {_friendly_api_error(exc)}") from exc


@plugin.mount_sandbox_method(SandboxMethodType.TOOL, name="GPT 单图编辑", description="使用 GPT 图像模型编辑一张图片")
async def gpt_image_edit(
    _ctx: AgentCtx,
    image_path: str,
    prompt: str,
    size: str = "",
    quality: Literal["auto", "low", "medium", "high"] = "auto",
    background: Literal["auto", "transparent", "opaque"] = "auto",
    send_to_chat: bool = True,
) -> str:
    """Edit one reference image with a GPT image model.

    Args:
        image_path: Sandbox image path supplied by the agent.
        prompt: Editing instruction.
        size: Optional size. Use "auto" or WIDTHxHEIGHT. Empty uses plugin default.
        quality: Optional quality: auto, low, medium, high.
        background: Optional background: auto, transparent, opaque.
        send_to_chat: Send the edited image to the current chat after generation.

    Returns:
        The edited image sandbox path.
    """
    return await gpt_image_multi_edit(
        _ctx,
        reference_images=[{"image_path": image_path, "description": "source image", "weight": 1.0}],
        target_prompt=prompt,
        size=size,
        quality=quality,
        background=background,
        send_to_chat=send_to_chat,
    )


@plugin.mount_sandbox_method(SandboxMethodType.TOOL, name="GPT 多图参考编辑", description="使用多张参考图片生成或编辑图片")
async def gpt_image_multi_edit(
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
        reference_images: List of references. Each item must contain image_path and may contain description and weight.
        target_prompt: Desired final image.
        size: Optional size. Use "auto" or WIDTHxHEIGHT. Empty uses plugin default.
        quality: Optional quality: auto, low, medium, high.
        background: Optional background: auto, transparent, opaque.
        send_to_chat: Send the result to the current chat after generation.

    Returns:
        The generated or edited image sandbox path.
    """
    try:
        final_size = size or config.DEFAULT_SIZE
        prompt = build_prompt_with_references(
            target_prompt=target_prompt,
            references=reference_images,
            size=final_size,
        )
        image_data = await _edit_raw_image(
            _ctx=_ctx,
            prompt=prompt,
            reference_images=reference_images,
            size=final_size,
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
    except (OpenAIImageAPIError, HTTPStatusError, RequestError, ValueError) as exc:
        logger.error(f"GPT 图像编辑失败: {_friendly_api_error(exc)}")
        raise Exception(f"GPT 图像编辑失败: {_friendly_api_error(exc)}") from exc


@plugin.mount_collect_methods()
async def collect_available_methods(_ctx: AgentCtx) -> List[Any]:
    return [gpt_image_generate, gpt_image_edit, gpt_image_multi_edit]


@plugin.mount_cleanup_method()
async def clean_up():
    """Clean up plugin resources."""
