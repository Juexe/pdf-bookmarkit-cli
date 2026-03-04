"""
Typer CLI 应用程序主入口
"""
import typer
import asyncio
import os
from datetime import datetime
import json
import base64
import time
from pathlib import Path
from dotenv import load_dotenv

from bookmarkit.pdf_processor import extract_toc_images, apply_bookmarks
from bookmarkit.vlm_client import VlmClient
from bookmarkit.models import BookmarkNode

app = typer.Typer(help="通过 VLM 自动提权目录并插入书签的 PDF 工具")

def parse_page_range(range_str: str) -> list[int]:
    """
    解析用户提供的页码范围字符串（如 "1-3"）到 0-based index 列表
    """
    pages = []
    parts = range_str.split(',')
    for part in parts:
        if '-' in part:
            start_str, end_str = part.split('-')
            start = int(start_str.strip())
            end = int(end_str.strip())
            # 转换为 0-based
            pages.extend(list(range(start - 1, end)))
        else:
            pages.append(int(part.strip()) - 1)
    return pages


def load_bookmarks_from_logs(log_dir: Path) -> list[BookmarkNode]:
    """
    扫描日志目录下所有 vlm_response_*.txt 文件，
    按文件名中的窗口编号排序后解析并合并去重。
    """
    import re
    import os

    os.environ.setdefault("VLM_MODEL", "offline")
    os.environ.setdefault("VLM_API_KEY", "offline")
    os.environ.setdefault("VLM_BASE_URL", "http://offline")

    client = VlmClient()

    pattern = re.compile(r"vlm_response_win_(\d+)_.*_attempt_\d+\.txt$")
    response_files = []
    for f in log_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            response_files.append((int(m.group(1)), f))

    if not response_files:
        raise FileNotFoundError(f"日志目录 {log_dir} 中没有找到任何 vlm_response_*.txt 文件")

    # 每个窗口只取编号最小的 attempt（attempt_1 优先）
    best: dict[int, Path] = {}
    attempt_pat = re.compile(r"_attempt_(\d+)\.txt$")
    for win_idx, f in response_files:
        attempt = int(attempt_pat.search(f.name).group(1))
        if win_idx not in best or attempt < best[win_idx][0]:
            best[win_idx] = (attempt, f)

    sorted_files = [best[win_idx][1] for win_idx in sorted(best.keys())]
    typer.echo(f"从日志目录找到 {len(sorted_files)} 个窗口响应文件")

    all_bookmarks: list[BookmarkNode] = []
    for f in sorted_files:
        text = f.read_text(encoding="utf-8")
        data = client._extract_json(text)
        bms = client._parse_bookmarks(data)
        typer.echo(f"  {f.name}: {len(bms)} 个顶级节点")
        all_bookmarks.extend(bms)

    merged = client._merge_and_deduplicate(all_bookmarks)
    typer.echo(f"合并去重完成：{len(merged)} 个顶级节点")
    return merged


@app.command()
def process(
    pdf_path: str = typer.Argument(None, help="原 PDF 的路径"),
    toc: str = typer.Option(None, "--toc", help="目录页码范围，如 '1-3' 或 '1,2,3' (1-based)"),
    first: int = typer.Option(None, "--first", help="PDF 第一页(正文阿拉伯数字页码为1的那一页) 在整个 PDF 文件中的实际页码 (1-based)"),
    output: str = typer.Option(None, "--output", "-o", help="输出 PDF 路径（默认在原目录生成 _bookmarked 文件）"),
    from_logs: str = typer.Option(None, "--from-logs", help="跳过 VLM 调用，直接读取指定日志目录下的 response 数据生成书签"),
):
    """
    处理 PDF，通过传入的目录页获取大模型生成的书签树，并应用到该 PDF 中。
    使用 --from-logs <日志目录> 可跳过 VLM 调用，直接复用已有的识别结果。
    """
    load_dotenv()

    # ── --from-logs 模式 ──────────────────────────────────────────────────
    if from_logs:
        log_dir = Path(from_logs)
        if not log_dir.is_dir():
            typer.secho(f"错误: 日志目录不存在: {log_dir}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        # 从 user_input.json 读取原始参数（允许命令行覆盖）
        user_input_path = log_dir / "user_input.json"
        if user_input_path.exists():
            with open(user_input_path, encoding="utf-8") as f:
                saved = json.load(f)
            pdf_path = pdf_path or saved.get("pdf_path")
            toc = toc or saved.get("toc_range")
            first = first or saved.get("first_page")
            output = output or saved.get("output")
            typer.echo(f"已从 user_input.json 读取参数")

        # 仍缺少必要参数时提示
        if pdf_path is None:
            pdf_path = typer.prompt("请输入PDF路径")
        if first is None:
            first = typer.prompt("请输入第一页页码", type=int)

        if not Path(pdf_path).exists():
            typer.secho(f"错误: 找不到文件 {pdf_path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        out_path = output
        if not out_path:
            p = Path(pdf_path)
            out_path = str(p.with_name(f"{p.stem}_bookmarked{p.suffix}"))

        start_time = time.time()
        typer.echo(f"--from-logs 模式：跳过 VLM，直接读取日志目录 {log_dir}")

        try:
            bookmarks = load_bookmarks_from_logs(log_dir)
        except Exception as e:
            typer.secho(f"读取日志响应失败: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        typer.echo("正在将书签应用到 PDF 文件中...")
        try:
            apply_bookmarks(pdf_path, bookmarks, first, out_path)
            elapsed_time = time.time() - start_time
            typer.secho(f"处理成功！结果已保存至: {out_path}", fg=typer.colors.GREEN)
            typer.secho(f"总耗时: {elapsed_time:.2f} 秒", fg=typer.colors.CYAN)
        except Exception as e:
            typer.secho(f"写入书签到 PDF 时发生错误: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        return

    # ── 正常 VLM 模式 ─────────────────────────────────────────────────────
    if pdf_path is None:
        pdf_path = typer.prompt("请输入PDF路径")
    if toc is None:
        toc = typer.prompt("请输入目录页范围")
    if first is None:
        first = typer.prompt("请输入第一页页码", type=int)

    start_time = time.time()

    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_stem = Path(pdf_path).stem if pdf_path else "unknown"
    log_dir = Path(os.getcwd()) / "logs" / f"{timestamp}_{pdf_stem}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录用户输入参数
    params = {
        "pdf_path": pdf_path,
        "toc_range": toc,
        "first_page": first,
        "output": output,
    }
    with open(log_dir / "user_input.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    if not Path(pdf_path).exists():
        typer.secho(f"错误: 找不到文件 {pdf_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
        
    out_path = output
    if not out_path:
        p = Path(pdf_path)
        out_path = str(p.with_name(f"{p.stem}_bookmarked{p.suffix}"))
        
    try:
        toc_pages = parse_page_range(toc)
    except Exception as e:
        typer.secho(f"解析参数 --toc '{toc}' 失败: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo("正在从 PDF 提取目录图像...")
    try:
        images = extract_toc_images(pdf_path, toc_pages)
        typer.echo(f"成功提取了 {len(images)} 张图片。")
        
        # 保存提取出的图片记录
        img_dir = log_dir / "images"
        img_dir.mkdir(exist_ok=True)
        for i, img in enumerate(images):
            try:
                img_data = base64.b64decode(img.data)
                with open(img_dir / f"toc_page_{i}.jpg", "wb") as f:
                    f.write(img_data)
            except Exception as e:
                typer.secho(f"警告: 写入日志图片 toc_page_{i}.jpg 失败: {e}", fg=typer.colors.YELLOW)
                
    except Exception as e:
        typer.secho(f"提取图片失败: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
        
    try:
        client = VlmClient(log_dir=str(log_dir))
    except Exception as e:
        typer.secho(f"初始化 VLM 客户端失败 (请检查 .env 是否配置): {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
        
    typer.echo(f"正在通过VLM({client.model})识别目录结构，请耐心等待...")
    
    try:
        bookmarks = asyncio.run(client.recognize_toc(images))
        typer.echo(f"识别完成，共获取到 {len(bookmarks)} 个顶级节点。")
    except Exception as e:
        typer.secho(f"VLM 在处理时发生错误: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
        
    typer.echo("正在将书签应用到 PDF 文件中...")
    try:
        apply_bookmarks(pdf_path, bookmarks, first, out_path)
        
        elapsed_time = time.time() - start_time
        typer.secho(f"处理成功！结果已保存至: {out_path}", fg=typer.colors.GREEN)
        typer.secho(f"总耗时: {elapsed_time:.2f} 秒", fg=typer.colors.CYAN)
    except Exception as e:
        typer.secho(f"写入书签到 PDF 时发生错误: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
