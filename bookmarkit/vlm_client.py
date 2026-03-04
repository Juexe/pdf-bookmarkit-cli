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
        max_retries: Optional[int] = None,
        log_dir: Optional[str] = None
    ):
        self.model = model or os.getenv("VLM_MODEL")
        self.api_key = api_key or os.getenv("VLM_API_KEY")
        self.base_url = base_url or os.getenv("VLM_BASE_URL")
        self.response_format = ResponseFormat.create_bookmark_schema()

        if not self.base_url or not self.model or not self.api_key:
            raise ValueError("VLM_BASE_URL, VLM_MODEL, VLM_API_KEY 必须提供或在环境变量中设置")
        
        self.max_retries = max_retries or int(os.getenv("MAX_RETRIES", "3"))
        self.log_dir = log_dir
        self.max_concurrency = int(os.getenv("VLM_MAX_CONCURRENCY", "1"))
        logger.info(f"VlmClient 初始化成功，模型：{self.model}，最大并发：{self.max_concurrency}")

    def build_system_prompt(self) -> str:
        return """你正在分析 PDF 文档中的目录 (TOC) 页面。
任务是从图片中提取包含层级结构的书签数据。

要求：
1. 提取目录中所有可见条目。
2. 保留层级结构，使用 level 属性（0 表示顶级，1 表示下一级，以此类推）。
3. 准确提取它们在目录中标示的页码（数字）。
4. 保留标题原有内容（如： "1.1 背景"），去除公式和特殊符号。
5. 将子项嵌套在 children 数组中。
6. 如果没有子项，children 必须为空数组 []。"""

    async def recognize_toc(self, images: List[ProcessedImage]) -> List[BookmarkNode]:
        total = len(images)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        print(f"\n共 {total} 页目录，最大并发数: {self.max_concurrency}")

        async def _task(i: int, img: ProcessedImage) -> List[BookmarkNode]:
            async with semaphore:
                print(f"\n>>> [页 {i + 1}/{total}] 开始识别...")
                result = await self._recognize_single_image(img, i, total)
                print(f"<<< [页 {i + 1}/{total}] 识别完成")
                return result

        tasks = [_task(i, img) for i, img in enumerate(images)]
        results = await asyncio.gather(*tasks)

        # 按页码顺序拼接结果
        all_bookmarks = []
        for bookmarks in results:
            all_bookmarks.extend(bookmarks)
        return all_bookmarks

    async def _recognize_single_image(self, img: ProcessedImage, page_index: int, total: int = 1) -> List[BookmarkNode]:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                user_content = [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img.format.lower()};base64,{img.data}"
                    }
                }]

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
                    "timeout": 600,
                    "stream": True,
                }
                
                if self.log_dir:
                    import copy
                    from pathlib import Path
                    log_messages = copy.deepcopy(messages)
                    for msg in log_messages:
                        if isinstance(msg.get("content"), list):
                            for item in msg["content"]:
                                if item.get("type") == "image_url":
                                    item["image_url"]["url"] = "<BASE64_IMAGE_DATA_OMITTED>"
                                    
                    req_log = {
                        "model": self.model,
                        "base_url": self.base_url,
                        "temperature": kwargs.get("temperature"),
                        "timeout": kwargs.get("timeout"),
                        "stream": kwargs.get("stream"),
                        "messages": log_messages
                    }
                    req_path = Path(self.log_dir) / f"vlm_request_page_{page_index + 1}_attempt_{attempt + 1}.json"
                    with open(req_path, "w", encoding="utf-8") as f:
                        json.dump(req_log, f, ensure_ascii=False, indent=2)

                tag = f"[页 {page_index + 1}/{total}]"
                logger.info(f"{tag} 第 {attempt + 1} 次请求 VLM (模型: {self.model})...")
                print(f"{tag} -> 正在发送图像至 VLM...")
                
                log_file = None
                
                if self.log_dir:
                    from pathlib import Path
                    resp_path = Path(self.log_dir) / f"vlm_response_page_{page_index + 1}_attempt_{attempt + 1}.txt"
                    log_file = open(resp_path, "w", encoding="utf-8")
                    # print(f"-> 响应正流式输出到日志，可打开文件实时查看最新进展: {resp_path}")

                response_stream = await acompletion(**kwargs)
                response_text = ""
                
                try:
                    char_count = 0
                    async for chunk in response_stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            response_text += content
                            char_count += len(content)
                            print(f"\r{tag} -> 已接收: {char_count} 字符...", end="", flush=True)
                            
                            if log_file:
                                log_file.write(content)
                                log_file.flush()
                    print()  # 换行，避免后续输出在同一行
                finally:
                    if log_file:
                        log_file.close()

                print(f"{tag} -> 收到完整响应，正在解析...")
                
                bookmarks_data = self._extract_json(response_text)
                return self._parse_bookmarks(bookmarks_data)
                
            except Exception as e:
                last_error = e
                logger.warning(f"第 {attempt + 1} 次尝试失败：{str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"处理第 {page_index + 1} 页时所有 VLM 请求均失败: {last_error}")

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
