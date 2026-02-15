"""JSON serialization helpers for complex objects."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles Pydantic models, datetimes, and enums."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def serialize_to_json(obj: Any, indent: int = 2) -> str:
    """Serialize any object to a JSON string with enhanced type support.

    Args:
        obj: Object to serialize.
        indent: JSON indentation level.

    Returns:
        JSON string representation.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=indent)
    return json.dumps(obj, cls=EnhancedJSONEncoder, indent=indent)


def deserialize_from_json(json_str: str) -> Any:
    """Deserialize a JSON string to a Python object.

    Args:
        json_str: JSON string to parse.

    Returns:
        Parsed Python object.
    """
    return json.loads(json_str)
