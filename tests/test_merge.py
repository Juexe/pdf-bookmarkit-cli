"""
使用真实 VLM response 日志文件测试 _merge_and_deduplicate 算法。
用例来源：logs/20260304_213421_丘维声《高等代数》学习指导书第二版上/
"""
import json
import os
import sys
from pathlib import Path

# 确保能导入 bookmarkit
sys.path.insert(0, str(Path(__file__).parent.parent))

from bookmarkit.vlm_client import VlmClient
from bookmarkit.models import BookmarkNode

# ──────────────────────────────────────────────
# 准备环境（不需要真实 VLM 连接）
# ──────────────────────────────────────────────
os.environ.setdefault("VLM_MODEL", "test")
os.environ.setdefault("VLM_API_KEY", "test")
os.environ.setdefault("VLM_BASE_URL", "http://test")

client = VlmClient()

# ──────────────────────────────────────────────
# 读取 response 文件
# ──────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs" / "20260304_215518_丘维声《高等代数》学习指导书第二版上"

RESPONSE_FILES = [
    LOG_DIR / "vlm_response_win_1_page_1+2_attempt_1.txt",
    LOG_DIR / "vlm_response_win_2_page_2+3_attempt_1.txt",
    LOG_DIR / "vlm_response_win_3_page_3+4_attempt_1.txt",
    LOG_DIR / "vlm_response_win_4_page_4+5_attempt_1.txt",
]


def load_bookmarks(path: Path) -> list[BookmarkNode]:
    text = path.read_text(encoding="utf-8")
    data = client._extract_json(text)
    return client._parse_bookmarks(data)


def print_tree(nodes: list[BookmarkNode], indent: int = 0):
    for n in nodes:
        prefix = "  " * indent
        print(f"{prefix}[L{n.level}] p{n.page_number:4d}  {n.title}")
        if n.children:
            print_tree(n.children, indent + 1)


# ──────────────────────────────────────────────
# 主测试逻辑
# ──────────────────────────────────────────────
def main():
    # 1. 加载各窗口结果
    all_bookmarks: list[BookmarkNode] = []
    for f in RESPONSE_FILES:
        bms = load_bookmarks(f)
        print(f"[{f.name}] 读取到 {len(bms)} 个顶级节点")
        all_bookmarks.extend(bms)

    flat_before = client._flatten_bookmarks(all_bookmarks)
    print(f"\n合并前（展平）共 {len(flat_before)} 条")

    # 2. 执行合并去重
    merged = client._merge_and_deduplicate(all_bookmarks)
    flat_after = client._flatten_bookmarks(merged)
    print(f"合并后（展平）共 {len(flat_after)} 条，顶级节点 {len(merged)} 个")

    # 3. 打印结果树供人工检查
    print("\n" + "=" * 60)
    print("合并去重后完整目录树：")
    print("=" * 60)
    print_tree(merged)

    # 4. 基本断言
    # 页码应严格递增（忽略相同页码的相邻兄弟节点）
    pages = [n.page_number for n in flat_after]
    assert pages == sorted(pages), f"页码未按升序排列: {pages}"

    # 无重复 (title, page_number)
    keys = [(n.title.strip(), n.page_number) for n in flat_after]
    assert len(keys) == len(set(keys)), f"仍存在重复条目: {[k for k in keys if keys.count(k) > 1]}"

    print("\n✅ 所有断言通过")


if __name__ == "__main__":
    main()
