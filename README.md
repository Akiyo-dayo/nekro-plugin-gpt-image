# Nekro Agent AI 生图插件

支持 **OpenAI / Gemini / Stable Diffusion** 多后端同时配置的 Nekro Agent 图片生成与编辑插件。

## 功能特性

- **多后端同时配置** — 三个后端各自独立模型组，可同时启用
- **斜杠命令** — `/na-gpt`、`/na-gemini`、`/na-sd` 直接指定后端生图
- **预设系统** — 文本预设（提示词模板）和图片预设（快速图生图参考图）
- **WebUI 管理页面** — 可视化上传、管理和删除预设
- **沙盒工具** — AI 可自动调用文生图、图片编辑、预设生图等功能
- 文生图、单图编辑、多图参考编辑

## 安装

将本插件目录复制到 Nekro Agent 的插件包目录：

```
plugins/packages/gpt_image
```

然后重启 Nekro Agent 或在插件管理器中重新加载。

## 配置

每个后端拥有独立的模型组配置，可以同时启用所有后端。

### 后端配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `DEFAULT_BACKEND` | AI 自动调用时的默认后端 | `openai` |
| `OPENAI_MODEL_GROUP` | OpenAI 后端的模型组 | 空（禁用） |
| `OPENAI_MODEL_NAME` | OpenAI 模型名 | `gpt-image-1` |
| `GEMINI_MODEL_GROUP` | Gemini 后端的模型组 | 空（禁用） |
| `GEMINI_MODEL_NAME` | Gemini 模型名 | `gemini-2.0-flash-preview-image-generation` |
| `SD_MODEL_GROUP` | SD WebUI 后端的模型组 | 空（禁用） |
| `SD_MODEL_NAME` | SD 模型覆盖（留空使用当前模型） | 空 |

### 通用配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `DEFAULT_SIZE` | 默认图片尺寸 | `1024x1024` |
| `OUTPUT_FORMAT` | 输出格式：`png`/`jpeg`/`webp` | `png` |
| `OUTPUT_COMPRESSION` | JPEG/WebP 压缩质量（1-100） | `100` |
| `TIMEOUT_SECONDS` | HTTP 超时秒数 | `300` |
| `MAX_REFERENCE_IMAGES` | 多图编辑最大参考图数 | `5` |

### OpenAI 专属配置

| 配置项 | 默认值 |
|--------|--------|
| `DEFAULT_QUALITY` | `auto` |
| `DEFAULT_BACKGROUND` | `auto` |
| `MODERATION` | `auto` |

### Stable Diffusion 专属配置

| 配置项 | 默认值 |
|--------|--------|
| `SD_NEGATIVE_PROMPT` | 空 |
| `SD_STEPS` | `20` |
| `SD_CFG_SCALE` | `7.0` |
| `SD_SAMPLER` | `Euler a` |
| `SD_DENOISING_STRENGTH` | `0.75` |

### 模型组配置示例

**OpenAI / 兼容 API：**
```yaml
MODEL_GROUPS:
  openai_image:
    BASE_URL: "https://api.openai.com/v1"
    API_KEY: "${OPENAI_API_KEY}"
    CHAT_MODEL: "gpt-image-1"
```

**Google Gemini：**
```yaml
MODEL_GROUPS:
  gemini_image:
    BASE_URL: "https://generativelanguage.googleapis.com/v1beta"
    API_KEY: "${GEMINI_API_KEY}"
    CHAT_MODEL: "gemini-2.0-flash-preview-image-generation"
```

**Stable Diffusion WebUI（本地）：**
```yaml
MODEL_GROUPS:
  sd_local:
    BASE_URL: "http://127.0.0.1:7860"
    API_KEY: ""
```

## 斜杠命令

| 命令 | 说明 |
|------|------|
| `/na-gpt <提示词>` | 使用 OpenAI 后端生图 |
| `/na-gemini <提示词>` | 使用 Gemini 后端生图 |
| `/na-sd <提示词>` | 使用 SD WebUI 后端生图 |
| `/na-preset.list` | 列出所有已保存的预设 |
| `/na-preset.add-text <名称> <模板>` | 添加文本预设 |
| `/na-preset.delete <名称>` | 删除预设 |

别名：`gpt画图`、`gemini画图`、`sd画图`

命令消息本身附带图片，或回复一条图片消息再执行 `/na-gpt`、`/na-gemini`、`/na-sd` 时，插件会自动把这些图片作为参考图，走对应后端的图生图/编辑链路。

## 预设系统

### 文本预设

文本预设是提示词模板，支持 `{input}` 占位符：

```
/na-preset.add-text 动漫风 将以下内容画成动漫风格: {input}
```

当命令的提示词匹配到预设名时，自动使用模板生成。

### 图片预设

图片预设存储一张参考图片，用于快速图生图。通过 **WebUI 管理页面**上传。

### WebUI 管理页面

插件提供可视化管理页面，访问地址：

```
http://<你的NA地址>/plugins/<插件key>/manage
```

在管理页面中可以：
- 📝 创建和管理文本预设（提示词模板、负面提示词、默认后端等）
- 🖼️ 上传和管理图片预设（拖拽上传参考图、设置默认提示词）
- 🗑️ 删除不需要的预设
- 查看当前已启用的后端和默认后端

## AI 可调用的沙盒工具

| 工具名 | 说明 |
|--------|------|
| `AI 文生图` | 从文字提示生成图片（使用默认后端） |
| `AI 单图编辑` | 编辑一张图片 |
| `AI 多图参考编辑` | 使用多张参考图生成或编辑图片 |
| `AI 预设生图` | 使用已保存的预设快速生成图片 |

## 运行测试

```bash
cd tests
python -m pytest test_units.py -v
```

## 说明

- `module_name` 保持 `gpt_image` 以向后兼容已有安装。
- 本仓库仅包含插件源代码，不包含运行时配置、数据和密钥文件。
