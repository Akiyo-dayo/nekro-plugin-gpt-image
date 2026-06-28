"""预设管理模块

管理图片预设（用于快速图生图）和文本预设（提示词模板）。
预设以 JSON 文件存储在插件数据目录下。
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ImagePreset(BaseModel):
    """图片预设 - 存储一张 base64 图片用于快速图生图"""
    name: str
    description: str = ""
    image_b64: str = Field(description="base64 encoded image data (no data URI prefix)")
    mime_type: str = "image/png"
    default_prompt: str = ""
    default_size: str = ""
    default_backend: str = ""
    created_at: float = Field(default_factory=time.time)


class TextPreset(BaseModel):
    """文本预设 - 存储提示词模板"""
    name: str
    description: str = ""
    prompt_template: str = Field(description="Prompt template; use {input} as placeholder for user input")
    negative_prompt: str = ""
    default_size: str = ""
    default_quality: str = "auto"
    default_backend: str = ""
    created_at: float = Field(default_factory=time.time)


class PresetStore:
    """预设持久化管理器"""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._image_dir = data_dir / "presets" / "images"
        self._text_dir = data_dir / "presets" / "texts"
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._text_dir.mkdir(parents=True, exist_ok=True)

    def _image_path(self, name: str) -> Path:
        return self._image_dir / f"{name}.json"

    def _text_path(self, name: str) -> Path:
        return self._text_dir / f"{name}.json"

    # -- Image presets --

    def save_image_preset(self, preset: ImagePreset) -> None:
        path = self._image_path(preset.name)
        path.write_text(preset.model_dump_json(indent=2), encoding="utf-8")

    def get_image_preset(self, name: str) -> Optional[ImagePreset]:
        path = self._image_path(name)
        if not path.exists():
            return None
        return ImagePreset.model_validate_json(path.read_text(encoding="utf-8"))

    def list_image_presets(self) -> List[Dict[str, Any]]:
        results = []
        for p in sorted(self._image_dir.glob("*.json")):
            try:
                preset = ImagePreset.model_validate_json(p.read_text(encoding="utf-8"))
                results.append({
                    "name": preset.name,
                    "description": preset.description,
                    "default_prompt": preset.default_prompt,
                    "default_backend": preset.default_backend,
                    "mime_type": preset.mime_type,
                    "created_at": preset.created_at,
                })
            except Exception:
                continue
        return results

    def delete_image_preset(self, name: str) -> bool:
        path = self._image_path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    # -- Text presets --

    def save_text_preset(self, preset: TextPreset) -> None:
        path = self._text_path(preset.name)
        path.write_text(preset.model_dump_json(indent=2), encoding="utf-8")

    def get_text_preset(self, name: str) -> Optional[TextPreset]:
        path = self._text_path(name)
        if not path.exists():
            return None
        return TextPreset.model_validate_json(path.read_text(encoding="utf-8"))

    def list_text_presets(self) -> List[Dict[str, Any]]:
        results = []
        for p in sorted(self._text_dir.glob("*.json")):
            try:
                preset = TextPreset.model_validate_json(p.read_text(encoding="utf-8"))
                results.append({
                    "name": preset.name,
                    "description": preset.description,
                    "prompt_template": preset.prompt_template,
                    "negative_prompt": preset.negative_prompt,
                    "default_backend": preset.default_backend,
                    "created_at": preset.created_at,
                })
            except Exception:
                continue
        return results

    def delete_text_preset(self, name: str) -> bool:
        path = self._text_path(name)
        if path.exists():
            path.unlink()
            return True
        return False