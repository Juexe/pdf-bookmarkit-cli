# PDF Bookmarkit CLI

`pdf-bookmarkit-cli` 是一个使用 Python, Typer 构建的命令行工具，用于直接从 PDF 提取目录页面，并通过调用大语言视觉模型 (VLM) 自动解析目录层级结构，最终将结构化树状书签注入回带有页码偏移补偿的 PDF 中。

## 功能特性
- **纯本地命令行操作**，脱离前后端依赖服务。
- 使用 `PyMuPDF` 自动完成高清晰度目录图片的提取和底层 PDF 书签注入。
- 支持对接 `LiteLLM` 提供任意兼容 OpenAI 接口标准的 Vision Language Model（如 Qwen, GPT-4o 等），实现高级的结构化数据提取 (`Structured Outputs`)。

## 快速运行

本项目使用 `uv` 管理和运行：

### 1. 准备环境
确保安装了 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
然后进入 `pdf-bookmarkit-cli` 目录，安装依赖：
```bash
uv sync
```

### 2. 配置 VLM 密钥
在 `pdf-bookmarkit-cli` 根目录下创建一个 `.env` 文件，并填写您的 VLM 信息：
```env
VLM_API_KEY="sk-xxxxxxxxxxx"
VLM_BASE_URL="https://api.example.com/v1"
VLM_MODEL="gpt-4o"
```

### 3. 使用命令行
使用 `uv run bookmarkit` 传入所需的 PDF 以及结构参数。

命令基本格式：
```bash
uv run bookmarkit <PDF文件路径> --toc "<目录起始和终止页码>" --first <正文第一页对应的绝对物理页码>
```

**示例**：
提取 `math_book.pdf` 第1到第3页作为目录图片交由 VLM 识别，然后将书签偏移映射为正文第5页是 PDF 的第1页：
```bash
uv run bookmarkit math_book.pdf --toc "1-3" --first 5
```

如果成功，您将在原文同目录下得到一个带有书签树的 `math_book_bookmarked.pdf` 副本文件。

## 详细参数参考
- `pdf_path` (必填/位置参数)：待处理的原 PDF 文件路径。
- `--toc` (必填/选项)：PDF的物理目录起始与结束页。请注意，这里的页码是人们习惯的 `1-based` (第一页就是1)。支持例如 `"1-3"` (代表1,2,3) 或者散装 `"1,2,3"`。
- `--first` (必填/选项)：文档真正的第一页 (阿拉伯数字为1的页面) 在未处理的 PDF 文件中的**绝对物理页数** (也是 1-based)。这用于校准 VLM 抽取的书签相对跳转。
- `--output` / `-o` (选填)：如果指定，将结果保存为您填写的专门路径。否则自动追加 `_bookmarked` 后缀。
