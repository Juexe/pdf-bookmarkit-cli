"""
数据模型定义，用于 PDF CLI 的书签结构和 VLM 响应格式化。
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any

class BookmarkNode(BaseModel):
    """
    表示 PDF 书签层级中的单个书签节点。
    """
    title: str = Field(
        ..., 
        min_length=1, 
        max_length=500,
        description="书签标题"
    )
    page_number: int = Field(
        ..., 
        ge=1,
        description="目标页码（至少为 1）"
    )
    level: int = Field(
        ..., 
        ge=0, 
        le=10,
        description="层级深度（0为顶级，以此向下）"
    )
    children: List['BookmarkNode'] = Field(
        default_factory=list,
        description="子书签节点列表"
    )
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("书签标题不能为空")
        return v


class ResponseFormat(BaseModel):
    """
    使用 Structured Outputs 的 VLM 响应格式。
    """
    type: str = Field(default="json_schema")
    json_schema: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create_bookmark_schema(cls) -> "ResponseFormat":
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "page_number": {"type": "integer"},
                    "level": {"type": "integer"},
                    "children": {
                        "type": "array",
                        "items": {"$ref": "#/items"}
                    }
                },
                "required": ["title", "page_number", "level", "children"],
                "additionalProperties": False
            }
        }
        return cls(
            type="json_schema",
            json_schema={
                "name": "bookmark_list",
                "strict": True,
                "schema": schema
            }
        )

class ProcessedImage(BaseModel):
    """
    处理后的图片数据，用于 VLM 识别。
    """
    data: str = Field(..., description="Base64 编码的图片数据")
    format: str = Field(..., description="图片格式 (例如 PNG 或 JPEG)")
    width: int = Field(..., gt=0, description="图片宽度")
    height: int = Field(..., gt=0, description="图片高度")
