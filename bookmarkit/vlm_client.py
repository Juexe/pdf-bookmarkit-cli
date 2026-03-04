"""
VLM Client 用于从 VLM 获取结构化的 PDF 目录数据。
"""
import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from litellm import acompletion
from bookmarkit.models import BookmarkNode, ProcessedImage, ResponseFormat

logger = logging.getLogger(__name__)

class VlmClient:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3
    ):
        self.model = model or os.getenv("VLM_MODEL")
        self.api_key = api_key or os.getenv("VLM_API_KEY")
        self.base_url = base_url or os.getenv("VLM_BASE_URL")
        self.response_format = ResponseFormat.create_bookmark_schema()

        if not self.base_url or not self.model or not self.api_key:
            raise ValueError("VLM_BASE_URL, VLM_MODEL, VLM_API_KEY 必须提供或在环境变量中设置")
        
        self.max_retries = max_retries
        logger.info(f"VlmClient 初始化成功，模型：{self.model}")

    def build_system_prompt(self) -> str:
        return """你正在分析 PDF 文档中的目录 (TOC) 页面。
任务是从图片中提取包含层级结构的书签数据。

要求：
1. 提取目录中所有可见条目。
2. 保留层级结构，使用 level 属性（0 表示顶级，1 表示下一级，以此类推）。
3. 准确提取它们在目录中标示的页码（数字）。
4. 保留标题原有格式（如： "1.1 背景"）。
5. 将子项嵌套在 children 数组中。
6. 如果没有子项，children 必须为空数组 []。"""

    async def recognize_toc(self, images: List[ProcessedImage]) -> List[BookmarkNode]:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                user_content = []
                for img in images:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img.format.lower()};base64,{img.data}"
                        }
                    })

                messages = [
                    {"role": "system", "content": self.build_system_prompt()},
                    {"role": "user", "content": user_content}
                ]

                kwargs = {
                    "base_url": self.base_url,
                    "model": self.model,
                    "messages": messages,
                    "api_key": self.api_key,
                    "temperature": 0.1,
                    "allowed_openai_params": ['response_format'],
                    "response_format": self.response_format.model_dump(),
                }

                logger.info(f"第 {attempt + 1} 次请求 VLM (模型: {self.model})...")
                print(f"-> 正在打包 {len(images)} 张提取图像并发送至 VLM...")
                print(f"-> 正在等待模型 {self.model} 解析响应，这可能需要几十秒的时间...")
                response = await acompletion(**kwargs)
                response_text = response.choices[0].message.content
                print("-> 收到 VLM 响应，正在解析结构...")
                
                bookmarks_data = self._extract_json(response_text)
                return self._parse_bookmarks(bookmarks_data)
                
            except Exception as e:
                last_error = e
                logger.warning(f"第 {attempt + 1} 次尝试失败：{str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"所有 VLM 请求均失败: {last_error}")

    def _extract_json(self, response_text: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(response_text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                return json.loads(response_text[start:end].strip())

        start = response_text.find("[")
        end = response_text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(response_text[start:end + 1])

        raise ValueError("无法从 VLM 响应提取 JSON 数组")

    def _parse_bookmarks(self, data: List[Dict[str, Any]]) -> List[BookmarkNode]:
        bookmarks = []
        for item in data:
            if "children" in item and item["children"]:
                item["children"] = self._parse_bookmarks(item["children"])
            bookmarks.append(BookmarkNode(**item))
        return bookmarks
