import base64
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from httpx import AsyncClient, Timeout


class OpenAIImageAPIError(RuntimeError):
    """Raised when the image API response does not contain a usable image."""


class GeminiImageAPIError(RuntimeError):
    """Raised when the Gemini image API response does not contain a usable image."""


class SDImageAPIError(RuntimeError):
    """Raised when the Stable Diffusion API response does not contain a usable image."""


ImageAPIError = (OpenAIImageAPIError, GeminiImageAPIError, SDImageAPIError)

_SIZE_PATTERN = re.compile(r"^\d{2,5}x\d{2,5}$")
_OUTPUT_FORMATS = {"png", "jpeg", "webp"}
_MIME_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


def normalize_size(size: str) -> str:
    value = (size or "auto").strip().lower()
    if value == "auto" or _SIZE_PATTERN.fullmatch(value):
        return value
    raise ValueError('size must be "auto" or WIDTHxHEIGHT, for example "1024x1024"')


def normalize_output_format(output_format: str) -> str:
    value = (output_format or "png").strip().lower()
    if value not in _OUTPUT_FORMATS:
        raise ValueError("output_format must be one of: png, jpeg, webp")
    return value


def image_file_name(output_format: str) -> str:
    return f"ai-image.{normalize_output_format(output_format)}"


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------

def build_generation_payload(
    *,
    model: str,
    prompt: str,
    size: str,
    quality: str,
    background: str,
    output_format: str,
    output_compression: int,
    moderation: str,
) -> Dict[str, Any]:
    fmt = normalize_output_format(output_format)
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "output_format": fmt,
    }

    normalized_size = normalize_size(size)
    if normalized_size != "auto":
        payload["size"] = normalized_size
    if quality != "auto":
        payload["quality"] = quality
    if background != "auto":
        payload["background"] = background
    if moderation != "auto":
        payload["moderation"] = moderation
    if fmt in {"jpeg", "webp"} and output_compression < 100:
        payload["output_compression"] = output_compression

    return payload


def decode_image_response(data: Mapping[str, Any], *, output_format: str) -> str:
    items = data.get("data")
    if isinstance(items, list) and items and isinstance(items[0], Mapping):
        first = items[0]
        b64_json = first.get("b64_json")
        if isinstance(b64_json, str) and b64_json.strip():
            fmt = normalize_output_format(output_format)
            return f"data:image/{fmt};base64,{b64_json.strip()}"

        image_url = first.get("url")
        if isinstance(image_url, str) and image_url.strip():
            return image_url.strip()

    raise OpenAIImageAPIError("No image was returned by the image API")


def decode_data_uri(data_uri: str) -> Tuple[str, bytes]:
    if not data_uri.startswith("data:") or "base64," not in data_uri:
        raise ValueError("invalid data URI image")
    header, encoded = data_uri.split("base64,", 1)
    mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    return mime_type, base64.b64decode(encoded.strip())


def extension_from_mime(mime_type: str) -> str:
    return _MIME_EXTENSIONS.get(mime_type.lower(), "bin")


def build_prompt_with_references(
    *,
    target_prompt: str,
    references: Sequence[Mapping[str, Any]],
    size: str,
) -> str:
    lines = [
        "Create a new image using the provided reference image(s).",
        f"Target: {target_prompt}",
        f"Requested size: {normalize_size(size)}",
        "",
        "Reference notes:",
    ]
    for index, reference in enumerate(references, 1):
        description = str(reference.get("description") or f"reference image {index}")
        weight = float(reference.get("weight", 1.0))
        suffix = f" (importance: {weight:g})" if weight != 1.0 else ""
        lines.append(f"Reference {index}: {description}{suffix}")
    lines.append("")
    lines.append("Preserve important visual traits from the references while following the target.")
    return "\n".join(lines)


def make_edit_files(image_items: Iterable[Tuple[str, str, bytes]]) -> List[Tuple[str, Tuple[str, bytes, str]]]:
    files: List[Tuple[str, Tuple[str, bytes, str]]] = []
    for index, (mime_type, name_hint, image_bytes) in enumerate(image_items, 1):
        extension = extension_from_mime(mime_type)
        safe_hint = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name_hint).strip("._") or f"reference_{index}"
        if "." not in safe_hint:
            safe_hint = f"{safe_hint}.{extension}"
        files.append(("image", (safe_hint, image_bytes, mime_type)))
    return files


async def post_generation(
    *,
    base_url: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout_seconds: int,
) -> Mapping[str, Any]:
    async with AsyncClient(timeout=Timeout(read=timeout_seconds, write=timeout_seconds, connect=10, pool=10)) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/images/generations",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=dict(payload),
        )
    response.raise_for_status()
    return response.json()


async def post_edit(
    *,
    base_url: str,
    api_key: str,
    fields: Mapping[str, Any],
    files: List[Tuple[str, Tuple[str, bytes, str]]],
    timeout_seconds: int,
) -> Mapping[str, Any]:
    async with AsyncClient(timeout=Timeout(read=timeout_seconds, write=timeout_seconds, connect=10, pool=10)) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/images/edits",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            data={key: str(value) for key, value in fields.items()},
            files=files,
        )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

def _gemini_size_config(size: str) -> Optional[Dict[str, int]]:
    """Convert WIDTHxHEIGHT to Gemini aspectRatio or personalization config."""
    normalized = normalize_size(size)
    if normalized == "auto":
        return None
    parts = normalized.split("x")
    return {"width": int(parts[0]), "height": int(parts[1])}


def build_gemini_generation_payload(
    *,
    prompt: str,
    size: str,
    quality: str,
    output_format: str,
    reference_images: Optional[List[Tuple[str, bytes]]] = None,
) -> Dict[str, Any]:
    """Build a Gemini generateContent request with responseModalities including image."""
    contents_parts: List[Dict[str, Any]] = []

    if reference_images:
        for mime_type, img_bytes in reference_images:
            contents_parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64.b64encode(img_bytes).decode(),
                }
            })

    contents_parts.append({"text": prompt})

    payload: Dict[str, Any] = {
        "contents": [{"parts": contents_parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }

    fmt = normalize_output_format(output_format)
    mime_map = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
    payload["generationConfig"]["responseMimeType"] = mime_map.get(fmt, "image/png")

    size_cfg = _gemini_size_config(size)
    if size_cfg:
        payload["generationConfig"]["imageSize"] = size_cfg

    return payload


def decode_gemini_response(data: Mapping[str, Any], *, output_format: str) -> str:
    """Extract base64 image from Gemini generateContent response."""
    candidates = data.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            inline = part.get("inlineData", {})
            b64_data = inline.get("data", "")
            mime_type = inline.get("mimeType", "")
            if b64_data and mime_type.startswith("image/"):
                return f"data:{mime_type};base64,{b64_data}"

    raise GeminiImageAPIError("No image was returned by the Gemini API")


async def post_gemini_generation(
    *,
    base_url: str,
    api_key: str,
    model: str,
    payload: Mapping[str, Any],
    timeout_seconds: int,
) -> Mapping[str, Any]:
    """Send a generateContent request to Gemini API."""
    url = f"{base_url.rstrip('/')}/models/{model}:generateContent"

    async with AsyncClient(timeout=Timeout(read=timeout_seconds, write=timeout_seconds, connect=10, pool=10)) as client:
        response = await client.post(
            url,
            headers={
                "Content-Type": "application/json",
            },
            params={"key": api_key},
            json=dict(payload),
        )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Stable Diffusion WebUI (A1111 / Forge) backend
# ---------------------------------------------------------------------------

def build_sd_txt2img_payload(
    *,
    prompt: str,
    negative_prompt: str = "",
    size: str,
    steps: int = 20,
    cfg_scale: float = 7.0,
    sampler_name: str = "Euler a",
    sd_model: str = "",
) -> Dict[str, Any]:
    """Build an SD WebUI /sdapi/v1/txt2img payload."""
    normalized = normalize_size(size)
    if normalized == "auto":
        width, height = 512, 512
    else:
        parts = normalized.split("x")
        width, height = int(parts[0]), int(parts[1])

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "n_iter": 1,
        "batch_size": 1,
    }

    if sd_model:
        payload["override_settings"] = {"sd_model_checkpoint": sd_model}

    return payload


def build_sd_img2img_payload(
    *,
    prompt: str,
    negative_prompt: str = "",
    init_images_b64: List[str],
    size: str,
    steps: int = 20,
    cfg_scale: float = 7.0,
    denoising_strength: float = 0.75,
    sampler_name: str = "Euler a",
    sd_model: str = "",
) -> Dict[str, Any]:
    """Build an SD WebUI /sdapi/v1/img2img payload."""
    normalized = normalize_size(size)
    if normalized == "auto":
        width, height = 512, 512
    else:
        parts = normalized.split("x")
        width, height = int(parts[0]), int(parts[1])

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "init_images": init_images_b64,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "denoising_strength": denoising_strength,
        "sampler_name": sampler_name,
        "n_iter": 1,
        "batch_size": 1,
    }

    if sd_model:
        payload["override_settings"] = {"sd_model_checkpoint": sd_model}

    return payload


def decode_sd_response(data: Mapping[str, Any], *, output_format: str) -> str:
    """Extract the first image from SD WebUI response."""
    images = data.get("images", [])
    if not images:
        raise SDImageAPIError("No image was returned by the Stable Diffusion API")

    b64_data = images[0]
    if isinstance(b64_data, str) and b64_data.strip():
        fmt = normalize_output_format(output_format)
        return f"data:image/{fmt};base64,{b64_data.strip()}"

    raise SDImageAPIError("Invalid image data returned by the Stable Diffusion API")


async def post_sd_txt2img(
    *,
    base_url: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout_seconds: int,
) -> Mapping[str, Any]:
    """Send a txt2img request to SD WebUI API."""
    headers: Dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with AsyncClient(timeout=Timeout(read=timeout_seconds, write=timeout_seconds, connect=10, pool=10)) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/sdapi/v1/txt2img",
            headers=headers,
            json=dict(payload),
        )
    response.raise_for_status()
    return response.json()


async def post_sd_img2img(
    *,
    base_url: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout_seconds: int,
) -> Mapping[str, Any]:
    """Send an img2img request to SD WebUI API."""
    headers: Dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with AsyncClient(timeout=Timeout(read=timeout_seconds, write=timeout_seconds, connect=10, pool=10)) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/sdapi/v1/img2img",
            headers=headers,
            json=dict(payload),
        )
    response.raise_for_status()
    return response.json()
