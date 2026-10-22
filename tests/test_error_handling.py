"""
SWYNEX Intelligent Feature 2 - Automated Test Suite: Error Handling & Edge Cases
Tests empty inputs, whitespace, None types, long text safety,
out-of-vocabulary gibberish, misleading keywords, and missing fields.
"""

import pytest
from src import validate_input
from src.classifier import classify_announcement
from src.priority_detector import detect_priority
from src.information_extractor import extract_information
from src.summarizer import summarize_announcement


class TestErrorHandlingAndValidation:
    """Tests rigorous edge case and error handling."""

    def test_empty_string_input(self):
        is_valid, cleaned, err, warn = validate_input("")
        assert is_valid is False
        assert "empty or contains only whitespace" in err

        clf_res = classify_announcement("")
        assert clf_res["category"] == "General"
        assert clf_res["confidence"] == 0.0

        prio_res = detect_priority("")
        assert prio_res["priority"] == "Low"

    def test_whitespace_only_input(self):
        is_valid, cleaned, err, warn = validate_input("   \n\t   ")
        assert is_valid is False
        assert err is not None

    def test_none_input(self):
        is_valid, cleaned, err, warn = validate_input(None)
        assert is_valid is False
        assert "cannot be None" in err

    def test_non_string_types(self):
        is_valid, cleaned, err, warn = validate_input(12345)
        assert is_valid is False
        assert "Expected string" in err

        is_valid, cleaned, err, warn = validate_input(["announcement"])
        assert is_valid is False

    def test_unusually_long_text_safety(self):
        # 10,000 character synthetic text
        massive_text = "Important university circular. " * 350
        is_valid, cleaned, err, warn = validate_input(massive_text, max_chars=5000)
        assert is_valid is True
        assert len(cleaned) == 5000
        assert warn is not None
        assert "Safely truncated" in warn

        # Verify pipeline handles safely without crash
        clf_res = classify_announcement(cleaned)
        assert "category" in clf_res
        sum_res = summarize_announcement(cleaned)
        assert "summary" in sum_res

    def test_brief_text_warning(self):
        short_text = "Hi students"
        is_valid, cleaned, err, warn = validate_input(short_text)
        assert is_valid is True
        assert warn is not None
        assert "very brief" in warn

    def test_unknown_gibberish_text(self):
        gibberish = "xyzabc qwerty12345 zxcvbnmasdfghjkl nonexistentterm123"
        clf_res = classify_announcement(gibberish)
        # Should not throw exception and should fallback gracefully
        assert clf_res["category"] in ["General", "Exam", "Assignment", "Event", "Placement", "Internship", "Holiday"]
        assert 0.0 <= clf_res["confidence"] <= 1.0

    def test_missing_deadlines_clean_message(self):
        text = "Curriculum guidelines and syllabus books are available in the departmental library."
        info = extract_information(text)
        assert len(info["deadlines"]) == 0
        assert "No explicit deadline" in info["status_messages"]["deadlines"]

    def test_misleading_urgency_keyword(self):
        # Word 'urgent' is part of 'urgent care clinic', not an administrative panic
        text = "The campus urgent care first-aid health clinic is located on ground floor near cafeteria."
        prio_res = detect_priority(text, category="General")
        # Should not crash and should explain reason clearly
        assert prio_res["priority"] in ["Low", "Medium", "High"]
        assert len(prio_res["reason"]) > 5

    def test_unicode_and_special_characters(self):
        text = "Notice: Seminar on AI & Quantum Computing — Room #404 at 10:30 AM! Fee: ₹500 / $10. 🚀"
        is_valid, cleaned, err, warn = validate_input(text)
        assert is_valid is True
        info = extract_information(cleaned)
        assert len(info["times"]) > 0
