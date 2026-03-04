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
    
    toc_list = []
    
    def traverse(nodes: List[BookmarkNode], current_level: int):
        for node in nodes:
            # 计算最终页码，PyMuPDF 中 set_toc() 要求的也是 1-based 页码。
            target_page = node.page_number + first_page_offset - 1
            # PyMuPDF 要求的书签项格式为一个 list: [层级(1-based), 标题, 页码(1-based), ...]
            toc_list.append([current_level, node.title, target_page])
            if node.children:
                traverse(node.children, current_level + 1)
                
    traverse(bookmarks, 1)
    
    doc.set_toc(toc_list)
    doc.save(output_path)
    doc.close()
