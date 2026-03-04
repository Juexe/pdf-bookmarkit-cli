# PDF Bookmarkit CLI

`pdf-bookmarkit-cli` 是一个使用 Python、Typer 构建的命令行工具，用于直接从 PDF 提取目录页面，并通过调用大语言视觉模型（VLM）自动解析目录层级结构，最终将结构化树状书签注入回带有页码偏移补偿的 PDF 中。

## 功能特性

- **纯本地命令行操作**，脱离前后端依赖服务。
- 使用 `PyMuPDF` 自动完成高清晰度目录图片的提取和底层 PDF 书签注入。
- 支持对接 `LiteLLM` 提供任意兼容 OpenAI 接口标准的 Vision Language Model（如 Qwen、GPT-4o、豆包等），实现结构化数据提取（`Structured Outputs`）。
- **滑动窗口识别**：每次将相邻两页同时发给 VLM 识别，提升跨页目录条目的识别准确性。
- **自动合并去重**：多窗口结果按页码排序并去重后，重建完整的嵌套书签树。
- **详细日志记录**：每次运行自动在 `logs/` 下创建带时间戳的子目录，保存请求参数、响应原文和缩略图。
- **`--from-logs` 离线重处理**：直接复用已有日志目录中的 VLM 响应，无需重新调用 VLM，适合调试和反复调整。

---

## 快速开始

### 1. 准备环境

确保已安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后在项目目录下执行：

```bash
uv sync
```

### 2. 配置 VLM 密钥

在项目根目录创建 `.env` 文件（可参考 `.env.example`）：

```env
VLM_API_KEY="sk-xxxxxxxxxxx"
VLM_BASE_URL="https://api.example.com/v1"
VLM_MODEL="gpt-4o"

# 失败重试次数（默认 1）
MAX_RETRIES=1

# 并发窗口数（默认 1 即串行，推荐按模型限速调整）
VLM_MAX_CONCURRENCY=4
```

**已知可用的火山引擎（豆包）模型：**
- `volcengine/doubao-seed-1-8-251228`
- `volcengine/doubao-seed-1-6-vision-250815`
- `volcengine/doubao-seed-1-6-flash-250828`

### 3. 运行

```bash
uv run bookmarkit [PDF文件路径] [--toc <目录页范围>] [--first <正文第一页物理页码>]
```

如果未提供必要参数，CLI 会自动交互提示输入。

**示例**：提取 `math_book.pdf` 第 1-3 页作为目录，正文第 1 页对应 PDF 物理第 5 页：

```bash
uv run bookmarkit math_book.pdf --toc "1-3" --first 5
```

执行成功后，将在原文件同目录下生成 `math_book_bookmarked.pdf`。

---

## 参数说明

| 参数 | 类型 | 说明 |
|---|---|---|
| `pdf_path` | 位置参数 | 待处理的原 PDF 文件路径，非必填（未提供时交互输入） |
| `--toc` | 选项 | 目录所在的物理页码范围（1-based），支持 `"1-3"` 或 `"1,2,3"` |
| `--first` | 选项 | 正文第一页（阿拉伯数字 1）在整个 PDF 中的绝对物理页码（1-based），用于校准书签页码偏移 |
| `--output` / `-o` | 选项 | 输出路径，默认在原文件同目录生成 `_bookmarked` 副本 |
| `--from-logs` | 选项 | 指定已有日志目录，跳过 VLM 调用，直接复用响应数据生成书签（见下文） |

---

## --from-logs：离线重处理

每次正常运行后，`logs/` 下会生成一个带时间戳的目录，结构如下：

```
logs/
└── 20260304_213421_math_book/
    ├── user_input.json                        # 本次运行的原始参数
    ├── images/                                # 提取的目录页缩略图
    ├── vlm_request_win_1_page_1+2_attempt_1.json
    ├── vlm_response_win_1_page_1+2_attempt_1.txt
    ├── vlm_request_win_2_page_2+3_attempt_1.json
    ├── vlm_response_win_2_page_2+3_attempt_1.txt
    └── ...
```

使用 `--from-logs` 可以跳过重新调用 VLM，直接复用这些响应文件：

```bash
uv run bookmarkit --from-logs logs/20260304_213421_math_book
```

`pdf_path`、`--first` 等参数会自动从目录内的 `user_input.json` 读取，也可以在命令行中手动覆盖：

```bash
# 覆盖输出路径
uv run bookmarkit --from-logs logs/20260304_213421_math_book -o my_output.pdf

# 覆盖 first_page（如上次填错了）
uv run bookmarkit --from-logs logs/20260304_213421_math_book --first 10
```

---

## 识别原理

```
PDF 文件
  ↓ PyMuPDF 提取目录页图片
图片列表 [p1, p2, p3, p4, p5]
  ↓ 滑动窗口分组
窗口1: [p1, p2] → VLM 识别 → 书签列表A
窗口2: [p2, p3] → VLM 识别 → 书签列表B
窗口3: [p3, p4] → VLM 识别 → 书签列表C
窗口4: [p4, p5] → VLM 识别 → 书签列表D
  ↓ 展平 + 去重（title+page_number 相同视为重复）+ 按页码排序 + 重建树
完整书签树
  ↓ PyMuPDF 注入 + 页码偏移校准
输出 PDF（含书签）
```
