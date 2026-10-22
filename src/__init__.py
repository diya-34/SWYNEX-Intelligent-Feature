"""
SWYNEX Intelligent Feature 2 - AI-Powered Student Announcement Intelligence System
Package initialization exposing primary analysis modules and input validation.
"""

from typing import Tuple, Optional, Any

from src.classifier import classify_announcement, AnnouncementClassifier, CATEGORIES
from src.priority_detector import detect_priority
from src.information_extractor import extract_information
from src.summarizer import summarize_announcement


def validate_input(text: Any, max_chars: int = 5000) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """
    Validates announcement text input with clear error and warning diagnostics.

    Returns:
        (is_valid: bool, cleaned_text: str, error_message: Optional[str], warning_message: Optional[str])
    """
    if text is None:
        return False, "", "Input text cannot be None. Please enter announcement content.", None

    if not isinstance(text, str):
        return False, "", f"Invalid input type: Expected string, got {type(text).__name__}.", None

    cleaned = text.strip()
    if not cleaned:
        return False, "", "Announcement text is empty or contains only whitespace. Please provide content to analyze.", None

    warning = None
    if len(cleaned) < 15:
        warning = f"Notice: Announcement text is very brief ({len(cleaned)} characters). Classification confidence may be limited."
    elif len(cleaned) > max_chars:
        warning = f"Notice: Announcement text exceeds {max_chars} characters ({len(cleaned)} chars). Safely truncated for analysis."
        cleaned = cleaned[:max_chars]

    return True, cleaned, None, warning


__version__ = "2.0.0"
__all__ = [
    "classify_announcement",
    "AnnouncementClassifier",
    "CATEGORIES",
    "detect_priority",
    "extract_information",
    "summarize_announcement",
    "validate_input",
]
