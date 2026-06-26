import base64
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from httpx import AsyncClient, Timeout


class OpenAIImageAPIError(RuntimeError):
    """Raised when the image API response does not contain a usable image."""


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
    return f"gpt-image.{normalize_output_format(output_format)}"


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

    raise OpenAIImageAPIError("No image was returned by the GPT image API")


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
