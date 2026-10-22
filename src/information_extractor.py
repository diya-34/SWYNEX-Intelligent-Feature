"""
SWYNEX Intelligent Feature 2 - Important Information Extraction Module
Uses regular expressions and rule-based heuristics to extract:
- Explicit dates and times
- Deadlines vs general dates
- Relative temporal indicators (tomorrow, tonight)
- URLs and website portals
- Contact email addresses
- Action keywords and entity tags
Provides clear fallback notices when information is not found.
"""

import re
from typing import Dict, Any, List, Optional


# Regular expressions for dates and temporal expressions
DATE_PATTERNS = [
    # Full Month Day, Year (e.g. October 25, 2026 / Oct 25, 2026 / 25th October 2026)
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:,?\s+\d{4})?\b",
    # DD/MM/YYYY or DD-MM-YYYY
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    # Day with time (e.g. Monday, November 10 at 9:00 AM / Friday 5:00 PM / Wednesday before 4 PM)
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:,?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2})?(?:\s+(?:at|before|from|by)\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))?\b",
]

# Time patterns (e.g. 11:59 PM, 5:00 PM, 9:00 AM, 12:00 PM)
TIME_PATTERN = r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)\b"

# Relative temporal indicators
RELATIVE_DATES = [
    "strictly tonight", "tonight", "tomorrow morning", "tomorrow afternoon",
    "tomorrow", "today", "next monday", "next wednesday", "next week",
    "this friday", "this saturday", "this thursday", "within 24 hours", "within 48 hours"
]

# URL pattern supporting http, https, www, and common academic/dev domain paths
URL_PATTERN = r"\b(?:https?://[^\s<>\"']+|www\.[^\s<>\"']+|[a-zA-Z0-9.-]+\.(?:edu|ac\.in|org|com|dev|io|gov\.in)(?:/[^\s<>\"']*)?)\b"

# Standard email pattern
EMAIL_PATTERN = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"

# Domain-specific action tags & keyword entities
ACTION_TAG_KEYWORDS = [
    "Admit Card", "Hall Ticket", "Seating Arrangement", "Late Fine", "Late Fee",
    "Turnitin", "Google Classroom", "Canvas LMS", "ERP Portal", "GitHub Repository",
    "Viva Voce", "Capstone Project", "Aptitude Test", "Technical Interview",
    "Mock Interview", "Resume Submission", "Bonafide Certificate", "Smart ID Card",
    "Scholarship", "Weather Advisory", "Biometric", "Debarment"
]


def extract_information(text: str) -> Dict[str, Any]:
    """
    Extracts structured entities from announcement text.

    Returns:
        Dictionary containing:
        - dates: list of identified date strings
        - deadlines: list of high-priority deadline strings
        - times: list of extracted time strings
        - urls: list of valid URLs
        - emails: list of valid email addresses
        - keywords: list of action tags detected
        - messages: user-friendly status messages for missing items
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "dates": [],
            "deadlines": [],
            "times": [],
            "urls": [],
            "emails": [],
            "keywords": [],
            "status_messages": {
                "dates": "No dates found (empty input).",
                "deadlines": "No deadlines found (empty input).",
                "urls": "No URLs found (empty input).",
                "emails": "No email addresses found (empty input).",
                "keywords": "No action keywords found (empty input).",
            }
        }

    raw_text = text.strip()
    text_lower = raw_text.lower()

    # 1. Extract Dates
    found_dates = []
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, raw_text, re.IGNORECASE)
        for m in matches:
            clean_m = m.strip(" ,.")
            if clean_m and clean_m not in found_dates and len(clean_m) > 2:
                found_dates.append(clean_m)

    # Check for relative dates
    for rel in RELATIVE_DATES:
        if re.search(r"\b" + re.escape(rel) + r"\b", text_lower):
            if rel.title() not in found_dates:
                found_dates.append(rel.title())

    # 2. Extract Times
    found_times = []
    time_matches = re.findall(TIME_PATTERN, raw_text)
    for tm in time_matches:
        if tm not in found_times:
            found_times.append(tm)

    # 3. Detect Deadlines vs General Dates
    # Words that signal a hard cutoff
    deadline_markers = ["deadline", "due date", "due by", "before", "until", "by", "closes on", "strictly tonight", "cutoff"]
    found_deadlines = []
    sentences = re.split(r"[.!?\n]+", raw_text)

    for sent in sentences:
        sent_lower = sent.lower()
        if any(marker in sent_lower for marker in deadline_markers):
            # Extract date/time from this specific sentence
            for d in found_dates:
                if d.lower() in sent_lower and d not in found_deadlines:
                    found_deadlines.append(d)
            for t in found_times:
                if t.lower() in sent_lower and t not in found_deadlines:
                    found_deadlines.append(t)

    # 4. Extract URLs
    found_urls = []
    for match in re.findall(URL_PATTERN, raw_text):
        clean_url = match.rstrip(".,;:)")
        if clean_url not in found_urls:
            found_urls.append(clean_url)

    # 5. Extract Emails
    found_emails = []
    for match in re.findall(EMAIL_PATTERN, raw_text):
        clean_email = match.lower().rstrip(".,;:)")
        if clean_email not in found_emails:
            found_emails.append(clean_email)

    # 6. Extract Action Tags
    found_keywords = []
    for tag in ACTION_TAG_KEYWORDS:
        if re.search(r"\b" + re.escape(tag) + r"\b", raw_text, re.IGNORECASE):
            if tag not in found_keywords:
                found_keywords.append(tag)

    # 7. Generate Friendly Status Messages for Missing Items
    status_messages = {
        "dates": "No specific dates found." if not found_dates else f"{len(found_dates)} date(s) found.",
        "deadlines": "No explicit deadline detected." if not found_deadlines else f"{len(found_deadlines)} deadline(s) identified.",
        "urls": "No URLs or web links found in this announcement." if not found_urls else f"{len(found_urls)} link(s) found.",
        "emails": "No contact email addresses found." if not found_emails else f"{len(found_emails)} email(s) found.",
        "keywords": "No specialized action tags detected." if not found_keywords else f"{len(found_keywords)} action tag(s) detected.",
    }

    return {
        "dates": found_dates,
        "deadlines": found_deadlines,
        "times": found_times,
        "urls": found_urls,
        "emails": found_emails,
        "keywords": found_keywords,
        "status_messages": status_messages,
    }
