"""Tests for client.py and presets.py — covers pure logic that can run without nekro_agent."""

import base64
import ast
import json
import tempfile
import shutil
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# client.py tests
# ---------------------------------------------------------------------------
from client import (
    OpenAIImageAPIError,
    GeminiImageAPIError,
    SDImageAPIError,
    normalize_size,
    normalize_output_format,
    image_file_name,
    build_generation_payload,
    decode_image_response,
    decode_data_uri,
    extension_from_mime,
    build_prompt_with_references,
    make_edit_files,
    build_gemini_generation_payload,
    decode_gemini_response,
    build_sd_txt2img_payload,
    build_sd_img2img_payload,
    decode_sd_response,
)


class TestNormalizeSize:
    def test_auto(self):
        assert normalize_size("auto") == "auto"
        assert normalize_size("") == "auto"
        assert normalize_size("  AUTO  ") == "auto"

    def test_valid_sizes(self):
        assert normalize_size("1024x1024") == "1024x1024"
        assert normalize_size("512x768") == "512x768"

    def test_invalid(self):
        with pytest.raises(ValueError):
            normalize_size("big")
        with pytest.raises(ValueError):
            normalize_size("1024")


class TestNormalizeOutputFormat:
    def test_valid(self):
        assert normalize_output_format("png") == "png"
        assert normalize_output_format("JPEG") == "jpeg"
        assert normalize_output_format("  WebP  ") == "webp"

    def test_default(self):
        assert normalize_output_format("") == "png"

    def test_invalid(self):
        with pytest.raises(ValueError):
            normalize_output_format("bmp")


class TestImageFileName:
    def test_names(self):
        assert image_file_name("png") == "ai-image.png"
        assert image_file_name("jpeg") == "ai-image.jpeg"


class TestBuildGenerationPayload:
    def test_minimal(self):
        p = build_generation_payload(
            model="m", prompt="test", size="auto", quality="auto",
            background="auto", output_format="png",
            output_compression=100, moderation="auto",
        )
        assert p["model"] == "m"
        assert p["prompt"] == "test"
        assert "size" not in p
        assert "quality" not in p

    def test_with_options(self):
        p = build_generation_payload(
            model="m", prompt="t", size="512x512", quality="high",
            background="transparent", output_format="jpeg",
            output_compression=80, moderation="low",
        )
        assert p["size"] == "512x512"
        assert p["quality"] == "high"
        assert p["output_compression"] == 80


class TestDecodeImageResponse:
    def test_b64(self):
        b64 = base64.b64encode(b"fake").decode()
        data = {"data": [{"b64_json": b64}]}
        result = decode_image_response(data, output_format="png")
        assert result.startswith("data:image/png;base64,")

    def test_url(self):
        data = {"data": [{"url": "https://example.com/img.png"}]}
        result = decode_image_response(data, output_format="png")
        assert result == "https://example.com/img.png"

    def test_empty_raises(self):
        with pytest.raises(OpenAIImageAPIError):
            decode_image_response({"data": []}, output_format="png")


class TestDecodeDataUri:
    def test_valid(self):
        b64 = base64.b64encode(b"hello").decode()
        uri = f"data:image/png;base64,{b64}"
        mime, data = decode_data_uri(uri)
        assert mime == "image/png"
        assert data == b"hello"

    def test_invalid(self):
        with pytest.raises(ValueError):
            decode_data_uri("not a data uri")


class TestBuildPromptWithReferences:
    def test_basic(self):
        refs = [{"description": "cat"}, {"description": "dog", "weight": 2.0}]
        prompt = build_prompt_with_references(
            target_prompt="merge", references=refs, size="1024x1024",
        )
        assert "cat" in prompt
        assert "dog" in prompt
        assert "importance: 2" in prompt


class TestMakeEditFiles:
    def test_creates_files(self):
        items = [("image/png", "test", b"data")]
        files = make_edit_files(items)
        assert len(files) == 1
        assert files[0][0] == "image"
        assert files[0][1][2] == "image/png"


class TestGeminiPayload:
    def test_text_only(self):
        p = build_gemini_generation_payload(
            prompt="hello", size="auto", quality="auto", output_format="png",
        )
        assert p["contents"][0]["parts"][-1]["text"] == "hello"
        assert "IMAGE" in p["generationConfig"]["responseModalities"]

    def test_with_refs(self):
        p = build_gemini_generation_payload(
            prompt="test", size="512x512", quality="high", output_format="jpeg",
            reference_images=[("image/png", b"fake")],
        )
        assert len(p["contents"][0]["parts"]) == 2
        assert "inlineData" in p["contents"][0]["parts"][0]


class TestDecodeGeminiResponse:
    def test_valid(self):
        b64 = base64.b64encode(b"img").decode()
        data = {"candidates": [{"content": {"parts": [
            {"inlineData": {"data": b64, "mimeType": "image/png"}}
        ]}}]}
        result = decode_gemini_response(data, output_format="png")
        assert result.startswith("data:image/png;base64,")

    def test_empty_raises(self):
        with pytest.raises(GeminiImageAPIError):
            decode_gemini_response({"candidates": []}, output_format="png")


class TestSDPayloads:
    def test_txt2img(self):
        p = build_sd_txt2img_payload(
            prompt="cat", size="512x512", steps=10, cfg_scale=7.0,
        )
        assert p["prompt"] == "cat"
        assert p["width"] == 512
        assert p["steps"] == 10

    def test_txt2img_auto_size(self):
        p = build_sd_txt2img_payload(prompt="cat", size="auto")
        assert p["width"] == 512
        assert p["height"] == 512

    def test_img2img(self):
        p = build_sd_img2img_payload(
            prompt="cat", init_images_b64=["abc"], size="768x768",
        )
        assert p["init_images"] == ["abc"]
        assert p["width"] == 768

    def test_decode_sd_response(self):
        b64 = base64.b64encode(b"img").decode()
        result = decode_sd_response({"images": [b64]}, output_format="png")
        assert result.startswith("data:image/png;base64,")

    def test_decode_sd_empty(self):
        with pytest.raises(SDImageAPIError):
            decode_sd_response({"images": []}, output_format="png")


# ---------------------------------------------------------------------------
# presets.py tests
# ---------------------------------------------------------------------------
from presets import ImagePreset, TextPreset, PresetStore, _safe_filename


class TestSafeFilename:
    def test_deterministic(self):
        assert _safe_filename("hello") == _safe_filename("hello")

    def test_different_names(self):
        assert _safe_filename("a") != _safe_filename("b")

    def test_length(self):
        long_name = "x" * 5000
        result = _safe_filename(long_name)
        assert len(result) == 64  # SHA-256 hex digest


class TestPresetStore:
    @pytest.fixture
    def store(self, tmp_path):
        return PresetStore(tmp_path)

    def test_text_preset_crud(self, store):
        preset = TextPreset(
            name="anime", prompt_template="draw {input} in anime style",
            description="Anime style",
        )
        store.save_text_preset(preset)
        loaded = store.get_text_preset("anime")
        assert loaded is not None
        assert loaded.prompt_template == "draw {input} in anime style"

        items = store.list_text_presets()
        assert len(items) == 1
        assert items[0]["name"] == "anime"

        assert store.delete_text_preset("anime") is True
        assert store.get_text_preset("anime") is None
        assert store.delete_text_preset("anime") is False

    def test_image_preset_crud(self, store):
        b64 = base64.b64encode(b"fake_image_data").decode()
        preset = ImagePreset(
            name="ref_cat", image_b64=b64, mime_type="image/png",
            default_prompt="a cute cat", description="Cat reference",
        )
        store.save_image_preset(preset)
        loaded = store.get_image_preset("ref_cat")
        assert loaded is not None
        assert loaded.image_b64 == b64

        items = store.list_image_presets()
        assert len(items) == 1
        assert items[0]["name"] == "ref_cat"

        assert store.delete_image_preset("ref_cat") is True
        assert store.get_image_preset("ref_cat") is None

    def test_missing_preset(self, store):
        assert store.get_text_preset("nope") is None
        assert store.get_image_preset("nope") is None

    def test_overwrite_preset(self, store):
        p1 = TextPreset(name="t", prompt_template="v1")
        store.save_text_preset(p1)
        p2 = TextPreset(name="t", prompt_template="v2")
        store.save_text_preset(p2)
        loaded = store.get_text_preset("t")
        assert loaded.prompt_template == "v2"

    def test_long_name_returns_none(self, store):
        """超长名称不应尝试文件操作，直接返回 None。"""
        long_name = "这是一段超级长的中文提示词" * 100
        assert store.get_text_preset(long_name) is None
        assert store.get_image_preset(long_name) is None
        assert store.delete_text_preset(long_name) is False
        assert store.delete_image_preset(long_name) is False

    def test_is_valid_preset_name(self):
        assert PresetStore.is_valid_preset_name("anime") is True
        assert PresetStore.is_valid_preset_name("") is False
        assert PresetStore.is_valid_preset_name("   ") is False
        assert PresetStore.is_valid_preset_name("x" * 100) is True
        assert PresetStore.is_valid_preset_name("x" * 101) is False

    def test_filenames_are_hashed(self, store, tmp_path):
        """确认文件名是哈希值而非原始名称。"""
        preset = TextPreset(name="测试中文名", prompt_template="test")
        store.save_text_preset(preset)
        text_dir = tmp_path / "presets" / "texts"
        files = list(text_dir.glob("*.json"))
        assert len(files) == 1
        filename = files[0].stem
        assert filename != "测试中文名"
        assert len(filename) == 64  # SHA-256

    def test_find_presets_skips_filesystem_for_long_prompt(self, tmp_path):
        """长提示词不应触发任何预设文件路径访问。"""

        class ExplodingPathStore(PresetStore):
            def _text_path(self, name: str) -> Path:
                raise AssertionError("text preset path should not be touched")

            def _image_path(self, name: str) -> Path:
                raise AssertionError("image preset path should not be touched")

        store = ExplodingPathStore(tmp_path)
        long_prompt = "这是一段很长的画图提示词" * 100
        text_preset, image_preset = store.find_presets(long_prompt)
        assert text_preset is None
        assert image_preset is None


class TestCommandImagePresetRouting:
    def test_image_preset_branch_uses_edit_pipeline(self):
        """图片预设分支必须走图生图/编辑链路，而不是纯文生图。"""
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        tree = ast.parse(plugin_path.read_text(encoding="utf-8"))
        cmd_generate = next(
            node for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_cmd_generate"
        )

        image_preset_branch = None
        for node in ast.walk(cmd_generate):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.BoolOp)
                and any(isinstance(value, ast.Name) and value.id == "image_preset" for value in node.test.values)
            ):
                image_preset_branch = node
                break

        assert image_preset_branch is not None
        call_names = [
            call.func.id
            for statement in image_preset_branch.body
            for call in ast.walk(statement)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        ]
        assert "_edit_preset_raw_image" in call_names
        assert "_generate_raw_image" not in call_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
