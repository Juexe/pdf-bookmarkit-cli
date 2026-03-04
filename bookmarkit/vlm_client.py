"""
VLM Client 用于从 VLM 获取结构化的 PDF 目录数据。
采用滑动窗口算法：每次将相邻两页同时发送给 VLM 识别，提升跨页识别准确性。
最后对所有窗口结果进行合并、去重、排序。
"""
import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
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

    def build_system_prompt(self, is_pair: bool = False) -> str:
        """构建系统提示词。is_pair=True 时说明发送了两页图片。"""
        base = """你正在分析 PDF 文档中的目录 (TOC) 页面。
任务是从图片中提取包含层级结构的书签数据。

要求：
1. 提取目录中所有可见条目。
2. 保留层级结构，使用 level 属性（0 表示顶级，1 表示下一级，以此类推）。
3. 准确提取它们在目录中标示的页码（数字）。
4. 保留标题原有内容（如： "1.1 背景"），去除公式和特殊符号。
5. 将子项嵌套在 children 数组中。
6. 如果没有子项，children 必须为空数组 []。"""

        if is_pair:
            base += """

注意：当前提供了两张连续的目录页图片。
- 第一张是前一页，第二张是后一页。
- 某些目录条目可能横跨两页（标题在前一页，页码在后一页），请完整识别此类条目。
- 请合并两页的所有目录条目，输出一个完整的结果列表，不要遗漏也不要重复。"""

        return base

    def _build_sliding_windows(self, images: List[ProcessedImage]) -> List[Tuple[int, ...]]:
        """
        构建滑动窗口分组。
        - 只有 1 页时返回 [(0,)]
        - 多页时返回 [(0,1), (1,2), (2,3), ...]
        """
        n = len(images)
        if n == 0:
            return []
        if n == 1:
            return [(0,)]
        return [(i, i + 1) for i in range(n - 1)]

    async def recognize_toc(self, images: List[ProcessedImage]) -> List[BookmarkNode]:
        """识别目录：滑动窗口发送相邻两页，最后合并去重。"""
        windows = self._build_sliding_windows(images)
        total_windows = len(windows)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        print(f"\n共 {len(images)} 页目录，构建 {total_windows} 个滑动窗口，最大并发数: {self.max_concurrency}")

        async def _task(win_idx: int, page_indices: Tuple[int, ...]) -> List[BookmarkNode]:
            async with semaphore:
                group_images = [images[i] for i in page_indices]
                page_label = "+".join(str(i + 1) for i in page_indices)
                print(f"\n>>> [窗口 {win_idx + 1}/{total_windows}] 页 {page_label} 开始识别...")
                result = await self._recognize_image_group(
                    group_images, win_idx, total_windows, page_indices
                )
                print(f"<<< [窗口 {win_idx + 1}/{total_windows}] 页 {page_label} 识别完成")
                return result

        tasks = [_task(i, win) for i, win in enumerate(windows)]
        results = await asyncio.gather(*tasks)

        # 合并去重排序
        all_bookmarks = []
        for bookmarks in results:
            all_bookmarks.extend(bookmarks)

        merged = self._merge_and_deduplicate(all_bookmarks)
        print(f"\n合并去重完成：窗口总计 {len(all_bookmarks)} 条 -> 去重后 {len(merged)} 条")
        return merged

    async def _recognize_image_group(
        self,
        imgs: List[ProcessedImage],
        window_index: int,
        total_windows: int,
        page_indices: Tuple[int, ...],
    ) -> List[BookmarkNode]:
        """发送一组图片（1~2 张）给 VLM 识别，带重试逻辑。"""
        is_pair = len(imgs) > 1
        last_error = None
        page_label = "+".join(str(i + 1) for i in page_indices)

        for attempt in range(self.max_retries):
            try:
                # 构建包含所有图片的 user content
                user_content = []
                for img in imgs:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img.format.lower()};base64,{img.data}"
                        }
                    })

                messages = [
                    {"role": "system", "content": self.build_system_prompt(is_pair)},
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
                        "is_pair": is_pair,
                        "page_indices": list(page_indices),
                        "messages": log_messages
                    }
                    req_path = Path(self.log_dir) / f"vlm_request_win_{window_index + 1}_page_{page_label}_attempt_{attempt + 1}.json"
                    with open(req_path, "w", encoding="utf-8") as f:
                        json.dump(req_log, f, ensure_ascii=False, indent=2)

                tag = f"[窗口 {window_index + 1}/{total_windows} 页{page_label}]"
                logger.info(f"{tag} 第 {attempt + 1} 次请求 VLM (模型: {self.model})...")
                print(f"{tag} -> 正在发送 {len(imgs)} 张图像至 VLM...")
                
                log_file = None
                
                if self.log_dir:
                    from pathlib import Path
                    resp_path = Path(self.log_dir) / f"vlm_response_win_{window_index + 1}_page_{page_label}_attempt_{attempt + 1}.txt"
                    log_file = open(resp_path, "w", encoding="utf-8")

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
                    print()  # 换行
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

        raise RuntimeError(f"处理窗口 {window_index + 1} (页 {page_label}) 时所有 VLM 请求均失败: {last_error}")

    def _flatten_bookmarks(self, nodes: List[BookmarkNode]) -> List[BookmarkNode]:
        """将嵌套的书签树展平为一维列表（保留 level 信息）。"""
        flat = []
        for node in nodes:
            # 创建不含 children 的副本
            flat.append(BookmarkNode(
                title=node.title,
                page_number=node.page_number,
                level=node.level,
                children=[]
            ))
            if node.children:
                flat.extend(self._flatten_bookmarks(node.children))
        return flat

    def _rebuild_tree(self, flat_nodes: List[BookmarkNode]) -> List[BookmarkNode]:
        """根据 level 信息将展平的书签列表重建为嵌套树结构。"""
        if not flat_nodes:
            return []

        root: List[BookmarkNode] = []
        # 栈中存储 (level, node) 对，用于追踪当前的层级路径
        stack: List[tuple[int, BookmarkNode]] = []

        for node in flat_nodes:
            new_node = BookmarkNode(
                title=node.title,
                page_number=node.page_number,
                level=node.level,
                children=[]
            )

            # 弹出栈中层级 >= 当前节点的所有项
            while stack and stack[-1][0] >= node.level:
                stack.pop()

            if not stack:
                # 没有父节点，这是顶级节点
                root.append(new_node)
            else:
                # 添加为栈顶节点的子节点
                stack[-1][1].children.append(new_node)

            stack.append((node.level, new_node))

        return root

    def _merge_and_deduplicate(self, bookmarks: List[BookmarkNode]) -> List[BookmarkNode]:
        """
        合并去重排序：
        1. 展平所有书签节点
        2. 按 (title, page_number) 去重，不论层级，保留首次出现的条目
        3. 按 page_number 排序（相同页码按在结果中首次出现的顺序）
        4. 重建嵌套树结构
        """
        flat = self._flatten_bookmarks(bookmarks)

        # 去重：相同 title + page_number 视为同一条目，不论 level
        seen = set()
        unique = []
        for node in flat:
            key = (node.title.strip(), node.page_number)
            if key not in seen:
                seen.add(key)
                unique.append(node)

        # 按页码排序（稳定排序，相同页码保持原序）
        unique.sort(key=lambda n: n.page_number)

        # 重建嵌套树
        return self._rebuild_tree(unique)

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
