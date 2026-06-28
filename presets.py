"""预设管理模块

管理图片预设（用于快速图生图）和文本预设（提示词模板）。
预设以 JSON 文件存储在插件数据目录下，文件名使用名称的 SHA-256 哈希值以避免文件名过长。
"""

import hashlib
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


def _safe_filename(name: str) -> str:
    """将预设名称转换为安全的文件名（SHA-256 哈希，固定 64 字符）。"""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


class PresetStore:
    """预设持久化管理器"""

    MAX_PRESET_NAME_LENGTH = 100

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._image_dir = data_dir / "presets" / "images"
        self._text_dir = data_dir / "presets" / "texts"
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._text_dir.mkdir(parents=True, exist_ok=True)

    def _image_path(self, name: str) -> Path:
        return self._image_dir / f"{_safe_filename(name)}.json"

    def _text_path(self, name: str) -> Path:
        return self._text_dir / f"{_safe_filename(name)}.json"

    @classmethod
    def is_valid_preset_name(cls, name: str) -> bool:
        """检查名称是否可能是预设名（长度合理）。"""
        return 0 < len(name.strip()) <= cls.MAX_PRESET_NAME_LENGTH

    # -- Image presets --

    def save_image_preset(self, preset: ImagePreset) -> None:
        path = self._image_path(preset.name)
        path.write_text(preset.model_dump_json(indent=2), encoding="utf-8")

    def get_image_preset(self, name: str) -> Optional[ImagePreset]:
        if not self.is_valid_preset_name(name):
            return None
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
        if not self.is_valid_preset_name(name):
            return False
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
        if not self.is_valid_preset_name(name):
            return None
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
        if not self.is_valid_preset_name(name):
            return False
        path = self._text_path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def find_presets(self, name: str) -> tuple[Optional[TextPreset], Optional[ImagePreset]]:
        """按名称查找文本/图片预设；长 prompt 会直接跳过，避免构造超长路径。"""
        if not self.is_valid_preset_name(name):
            return None, None
        return self.get_text_preset(name), self.get_image_preset(name)
