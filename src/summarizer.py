"""
SWYNEX Intelligent Feature 2 - Intelligent Summarizer Module
Implements a local, lightweight extractive summarization algorithm.
Extracts the most salient sentences based on frequency weighting,
positional importance, and action-verb relevance.
Guarantees ZERO hallucination because every output word originates verbatim from the input.
"""

import re
from typing import Dict, Any, List


# English stop words list for local processing without external downloads
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "won't", "would", "wouldn't", "you", "your",
    "yours", "yourself", "yourselves"
}

# High-information action signals that indicate critical instructions
ACTION_SIGNALS = [
    "must", "submit", "deadline", "mandatory", "required", "scheduled",
    "announced", "closes", "due", "registered", "report", "strictly", "portal"
]


def summarize_announcement(text: str, max_sentences: int = 2) -> Dict[str, Any]:
    """
    Generates a concise extractive summary of long announcement text.

    Args:
        text: Announcement text.
        max_sentences: Maximum number of sentences to include in summary.

    Returns:
        Dictionary containing:
        - summary: Extracted concise summary string
        - original_chars: Character count of original text
        - summary_chars: Character count of summary text
        - compression_ratio: Percentage reduction
        - sentences_selected: Number of sentences in summary
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "summary": "No content provided to summarize.",
            "original_chars": 0,
            "summary_chars": 0,
            "compression_ratio": "0.0%",
            "sentences_selected": 0,
        }

    raw_text = text.strip()
    original_len = len(raw_text)

    # Split text into sentences using punctuation boundaries
    sentence_candidates = re.split(r"(?<=[.!?])\s+", raw_text)
    sentences = [s.strip() for s in sentence_candidates if s.strip()]

    # If already concise (1 or 2 sentences), return directly
    if len(sentences) <= max_sentences:
        return {
            "summary": raw_text,
            "original_chars": original_len,
            "summary_chars": original_len,
            "compression_ratio": "0.0% (Text already brief)",
            "sentences_selected": len(sentences),
        }

    # Compute word frequency distribution
    words = re.findall(r"\b[a-zA-Z]{3,}\b", raw_text.lower())
    freq_table: Dict[str, int] = {}
    for word in words:
        if word not in STOP_WORDS:
            freq_table[word] = freq_table.get(word, 0) + 1

    max_freq = max(freq_table.values()) if freq_table else 1

    # Score each sentence
    sentence_scores: List[Dict[str, Any]] = []
    for idx, sentence in enumerate(sentences):
        sent_words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
        if not sent_words:
            continue

        # Base word frequency score
        raw_score = sum(freq_table.get(w, 0) for w in sent_words if w not in STOP_WORDS)
        normalized_score = raw_score / (len(sent_words) + 1e-5)

        # 1. Lead bias: First sentence in administrative notices almost always carries topic
        if idx == 0:
            normalized_score *= 1.35
        elif idx == 1:
            normalized_score *= 1.15

        # 2. Action Signal Bonus
        sent_lower = sentence.lower()
        if any(sig in sent_lower for sig in ACTION_SIGNALS):
            normalized_score *= 1.25

        # 3. Penalize extremely short sentences (e.g., "Thank you.")
        if len(sent_words) < 5:
            normalized_score *= 0.5

        sentence_scores.append({
            "index": idx,
            "sentence": sentence,
            "score": normalized_score,
        })

    if not sentence_scores:
        return {
            "summary": sentences[0] if sentences else raw_text,
            "original_chars": original_len,
            "summary_chars": len(sentences[0]) if sentences else original_len,
            "compression_ratio": "0.0%",
            "sentences_selected": 1,
        }

    # Sort by score descending and take top max_sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x["score"], reverse=True)[:max_sentences]

    # Reorder according to their original sequence in the text
    ordered_sentences = sorted(top_sentences, key=lambda x: x["index"])

    summary_text = " ".join(s["sentence"] for s in ordered_sentences)
    summary_len = len(summary_text)

    reduction = ((original_len - summary_len) / original_len) * 100 if original_len > 0 else 0
    reduction = max(0.0, reduction)

    return {
        "summary": summary_text,
        "original_chars": original_len,
        "summary_chars": summary_len,
        "compression_ratio": f"{reduction:.1f}%",
        "sentences_selected": len(ordered_sentences),
    }
