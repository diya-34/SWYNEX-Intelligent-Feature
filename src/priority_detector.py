"""
SWYNEX Intelligent Feature 2 - Priority Detection Module
Analyzes urgency, time sensitivity, deadlines, penalties, and key action terms
to assign High, Medium, or Low priority with an auditable, transparent reason.
"""

import re
from typing import Dict, Any, List, Optional


# Keywords signaling immediate or critical urgency (supports singular & plural)
HIGH_URGENCY_KEYWORDS = [
    r"urgent(?! care)", r"urgently", r"immediately", r"immediate", r"strictly tonight",
    r"today", r"tonight", r"tomorrow", r"within 24 hours", r"within 48 hours",
    r"last date", r"final call", r"last chance", r"without fail", r"strictly compulsory",
    r"compulsory", r"mandatory", r"hall tickets?", r"admit cards?",
    r"debarment", r"debarred", r"late fines?", r"late fees?", r"zero marks",
    r"penalt(y|ies)", r"strict compliance", r"evacuated", r"emergency closure",
    r"cyclone alert", r"severe", r"shortlisted candidates?", r"before friday"
]

# Keywords signaling moderate urgency / standard deadlines
MEDIUM_URGENCY_KEYWORDS = [
    r"deadline", r"due date", r"submit before", r"registration forms?", r"apply before",
    r"upcoming", r"schedule", r"commence", r"closes on", r"viva voce",
    r"mid-semester", r"backlog", r"re-evaluation", r"interviews?", r"assessments?",
    r"hackathon", r"workshop", r"fees?", r"dues?", r"prior notice", r"within a week",
    r"stipend of", r"training session", r"mock interview", r"closed on monday"
]

# Keywords signaling low urgency / informational notices
LOW_URGENCY_KEYWORDS = [
    r"for your information", r"fyi", r"optional", r"general notice", r"library hours",
    r"biometric", r"smart cards?", r"wi-fi maintenance", r"lost and found",
    r"canteen", r"bus concession", r"theme announced", r"winter break",
    r"annual sports", r"guest lecture", r"auditions?", r"urgent care",
    r"in due course", r"check back soon", r"elective courses"
]


def detect_priority(
    text: str,
    category: str = "General",
    extracted_dates: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Evaluates urgency and assigns a priority level (High, Medium, Low)
    along with an auditable explanation.

    Args:
        text: Raw announcement text.
        category: Classified category (e.g. Exam, Assignment).
        extracted_dates: Optional list of extracted temporal objects from information_extractor.

    Returns:
        Dictionary with priority ('High', 'Medium', 'Low'), score (0-100),
        reason, urgency_factors, and suggested action.
    """
    if not text or not text.strip():
        return {
            "priority": "Low",
            "score": 0,
            "reason": "Announcement text is empty; defaulted to Low priority.",
            "urgency_factors": [],
            "recommended_action": "Provide announcement text to analyze priority."
        }

    text_lower = text.lower()
    score = 25  # Baseline neutral score
    factors: List[str] = []

    # 1. Check High Urgency Keywords
    matched_high = []
    for pattern in HIGH_URGENCY_KEYWORDS:
        if re.search(r"\b" + pattern + r"\b", text_lower):
            matched_high.append(pattern.replace(r"\b", "").replace("?", ""))

    if matched_high:
        score += min(len(matched_high) * 20, 50)
        factors.append(f"Critical urgency terms: {', '.join(repr(w) for w in matched_high[:3])}")

    # 2. Check Medium Urgency Keywords
    matched_medium = []
    for pattern in MEDIUM_URGENCY_KEYWORDS:
        if re.search(r"\b" + pattern + r"\b", text_lower):
            matched_medium.append(pattern.replace(r"\b", "").replace("?", ""))

    if matched_medium:
        score += min(len(matched_medium) * 10, 25)
        factors.append(f"Action/deadline indicators: {', '.join(repr(w) for w in matched_medium[:3])}")

    # 3. Check Low Urgency Keywords
    matched_low = []
    for pattern in LOW_URGENCY_KEYWORDS:
        if re.search(r"\b" + pattern + r"\b", text_lower):
            matched_low.append(pattern.replace(r"\b", "").replace("?", ""))

    if matched_low and not matched_high:
        score -= min(len(matched_low) * 15, 25)
        factors.append(f"Informational/low urgency cues: {', '.join(repr(w) for w in matched_low[:2])}")

    # 4. Contextual Category Influence
    if category in ["Exam", "Placement"]:
        # Only boost if it is not just training or routine info
        if "training session" in text_lower or "mock interview" in text_lower:
            score += 5
        elif matched_high or "admit card" in text_lower or "timetable" in text_lower:
            score += 20
            factors.append(f"High-stakes category '{category}' with action requirement elevates priority.")
        else:
            score += 10
    elif category == "Assignment":
        if "due tomorrow" in text_lower or "zero marks" in text_lower or "tonight" in text_lower:
            score += 20
        else:
            score += 10
    elif category in ["Holiday", "General"]:
        if "emergency" in text_lower or "cyclone" in text_lower or "evacuated" in text_lower:
            score += 35
        else:
            score -= 10

    # 5. Temporal Proximity Boost
    if any(k in text_lower for k in ["strictly tonight", "within 24 hours", "evacuated today"]):
        score += 25
        factors.append("Imminent deadline within 24-48 hours detected.")

    # Clamp score between 0 and 100
    score = max(0, min(100, score))

    # Determine final Priority Level & Craft Clear Explanation
    if score >= 65:
        priority = "High"
        reason = "Assigned High priority because "
        if matched_high:
            clean_terms = [t for t in matched_high[:2] if "urgent" in t or "ticket" in t or "card" in t or "cyclone" in t or "zero" in t or "shortlist" in t]
            if clean_terms:
                reason += f"it contains critical urgency terms ({', '.join(clean_terms)}) "
            else:
                reason += "it contains critical urgency indicators "
        if any(k in text_lower for k in ["today", "tonight", "tomorrow", "strictly tonight", "24 hours"]):
            reason += "with an imminent cutoff/deadline. "
        else:
            reason += "and carries compulsory requirements or significant penalties. "
        action = "Immediate action required today. Verify requirements and complete prerequisites."

    elif score >= 35:
        priority = "Medium"
        reason = "Assigned Medium priority because "
        if matched_medium:
            reason += f"it specifies a formal deadline or submission window ({', '.join(matched_medium[:2])}). "
        else:
            reason += "it involves an upcoming academic task requiring planned action within the week. "
        action = "Schedule time in your calendar to complete this task before the indicated date."

    else:
        priority = "Low"
        reason = "Assigned Low priority as this is primarily an informational announcement with no urgent deadlines or mandatory penalties."
        action = "Read and archive for reference; no urgent action required."

    return {
        "priority": priority,
        "score": score,
        "reason": reason.strip(),
        "urgency_factors": factors,
        "recommended_action": action
    }
