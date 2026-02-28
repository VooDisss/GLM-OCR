"""Shared helpers for the pipeline package."""

from __future__ import annotations

from typing import Any, Dict, List

from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


def extract_image_urls(request_data: Dict[str, Any]) -> List[str]:
    """Extract image URLs from an OpenAI-style request payload."""
    image_urls: List[str] = []
    for msg in request_data.get("messages", []):
        if msg.get("role") == "user":
            contents = msg.get("content", [])
            if isinstance(contents, list):
                for content in contents:
                    if content.get("type") == "image_url":
                        image_urls.append(content["image_url"]["url"])
    return image_urls


def make_original_inputs(image_urls: List[str]) -> List[str]:
    """Strip ``file://`` prefix so that original paths are returned."""
    return [(url[7:] if url.startswith("file://") else url) for url in image_urls]


def extract_ocr_content(response: Dict[str, Any]) -> str:
    """Pull the content string out of an OpenAI-style OCR response."""
    return (
        response.get("choices", [{}])[0].get("message", {}).get("content", "")
    )


# ── Queue message "identifier" field values ──────────────────────────
# Every queue message is a dict with at least an "identifier" key.
IDENTIFIER_IMAGE = "image"
IDENTIFIER_REGION = "region"
IDENTIFIER_DONE = "done"
IDENTIFIER_ERROR = "error"
