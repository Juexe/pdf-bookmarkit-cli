"""
使用 PyMuPDF 进行 PDF 操作。
"""
import io
import fitz
import base64
from PIL import Image
from bookmarkit.models import ProcessedImage, BookmarkNode
from typing import List

def extract_toc_images(pdf_path: str, page_numbers: List[int]) -> List[ProcessedImage]:
    """
    将 PDF 指定的页码提取为图片，专为 VLM 读取进行了优化（缩放与 JPEG 压缩）。
    注意：这里的 page_numbers 是 0-based，如果用户输入的是 1, 也就是第一页，传进来应该是 0。
    """
    doc = fitz.open(pdf_path)
    images = []
    
    max_dim = 1024
    
    for p_num in page_numbers:
        page = doc.load_page(p_num)
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # 优化大小
        if img.width > max_dim or img.height > max_dim:
            scale = min(max_dim / img.width, max_dim / img.height)
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
            
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        images.append(ProcessedImage(
            data=b64,
            format="JPEG",
            width=img.width,
            height=img.height
        ))
        
    doc.close()
    return images


def apply_bookmarks(pdf_path: str, bookmarks: List[BookmarkNode], first_page_offset: int, output_path: str):
    """
    将识别出的书签层级注入到 PDF 副本中。
    """
    doc = fitz.open(pdf_path)
    
    flat_nodes = []
    
    def flatten(nodes: List[BookmarkNode]):
        for node in nodes:
            flat_nodes.append(node)
            if node.children:
                flatten(node.children)
                
    flatten(bookmarks)
    
    toc_list = []
    current_pymupdf_level = 0
    
    for node in flat_nodes:
        target_page = node.page_number + first_page_offset - 1
        
        # VLM 指定的 level，0是顶级
        desired_level = node.level + 1
        
        # PyMuPDF 要求：下一项的 level 不能比上一项的 level 大超过 1
        if desired_level > current_pymupdf_level + 1:
            desired_level = current_pymupdf_level + 1
            # 对于第一项，如果不从 1 开始，强制设为 1
            if current_pymupdf_level == 0:
                desired_level = 1
                
        # 允许 level 回退（如从 3 回到 1），这表示返回上级目录
        # 同级目录 level 不变
        toc_list.append([desired_level, node.title, target_page])
        current_pymupdf_level = desired_level
    
    doc.set_toc(toc_list)
    doc.save(output_path)
    doc.close()
