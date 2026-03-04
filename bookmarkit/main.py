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

@app.command()
def process(
    pdf_path: str = typer.Argument(None, help="原 PDF 的路径"),
    toc: str = typer.Option(None, "--toc", help="目录页码范围，如 '1-3' 或 '1,2,3' (1-based)"),
    first: int = typer.Option(None, "--first", help="PDF 第一页(正文阿拉伯数字页码为1的那一页) 在整个 PDF 文件中的实际页码 (1-based)"),
    output: str = typer.Option(None, "--output", "-o", help="输出 PDF 路径（默认在原目录生成 _bookmarked 文件）")
):
    """
    处理 PDF，通过传入的目录页获取大模型生成的书签树，并应用到该 PDF 中。
    """
    load_dotenv()  # 确保加载了当前目录或系统目录中的 .env 文件

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

    typer.echo(f"正在准备处理 PDF: {pdf_path}")
    typer.echo(f"日志将保存在: {log_dir}")
    
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

    typer.echo(f"将提取的物理页码 (0-based) 为: {toc_pages}")
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
        
    typer.echo("正在通过 VLM 识别目录结构，请耐心等待...")
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
