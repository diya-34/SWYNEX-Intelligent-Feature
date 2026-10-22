"""
SWYNEX Intelligent Feature 2 - Announcement Classification Module
Implements machine-learning and transparent rule-based text classification
into academic categories: Exam, Assignment, Event, Placement, Internship, Holiday, General.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


# Supported target categories
CATEGORIES: List[str] = [
    "Exam",
    "Assignment",
    "Event",
    "Placement",
    "Internship",
    "Holiday",
    "General",
]

# Auditable keyword baseline map for transparent classification & cold-start fallback
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Exam": [
        "exam", "examination", "timetable", "time table", "hall ticket", "admit card",
        "seating arrangement", "re-evaluation", "re-checking", "backlog", "supplementary",
        "mid-term", "mid-semester", "end-term", "end-semester", "viva voce", "viva",
        "invigilation", "answer script", "remedial test", "scribe", "debarment"
    ],
    "Assignment": [
        "assignment", "homework", "problem set", "submission", "submit", "turnitin",
        "plagiarism", "lab report", "project report", "github repository", "deliverable",
        "code submission", "canvas", "classroom", "lms", "due date", "mini-project"
    ],
    "Event": [
        "event", "hackathon", "workshop", "seminar", "webinar", "guest lecture",
        "auditions", "cultural fest", "tech fest", "symposium", "conference",
        "competition", "sports meet", "tournament", "auditorium", "tedx", "rhythm"
    ],
    "Placement": [
        "placement", "recruitment", "campus drive", "job offer", "shortlisted",
        "ctc", "lpa", "aptitude test", "coding assessment", "technical interview",
        "hr interview", "cdc", "placement cell", "tcs", "microsoft", "google hiring",
        "amazon hiring", "deloitte", "goldman sachs"
    ],
    "Internship": [
        "internship", "stipend", "summer intern", "winter intern", "industrial training",
        "trainee", "research internship", "intern", "iisc intern", "drdo intern",
        "gsoc", "remote internship", "ugrip"
    ],
    "Holiday": [
        "holiday", "closed", "closure", "suspended", "vacation", "break",
        "dussehra", "diwali", "eid", "christmas", "public holiday", "heavy rainfall",
        "cyclone alert", "weather advisory", "emergency closure", "classes resume"
    ],
    "General": [
        "notice", "circular", "library", "mess fee", "hostel", "id card", "smart card",
        "wifi", "wi-fi", "maintenance", "lost and found", "bonafide certificate",
        "parking sticker", "scholarship", "administrative", "canteen", "bus concession"
    ],
}


class AnnouncementClassifier:
    """
    Hybrid Machine-Learning and Rule-Based Announcement Classifier.
    Uses TF-IDF + Logistic Regression with calibrated confidence and
    transparent keyword fallback for resilient cold-start operation.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else self._get_default_model_path()
        self.pipeline: Optional[Pipeline] = None
        self._load_or_train()

    def _get_default_model_path(self) -> Path:
        base_dir = Path(__file__).resolve().parent.parent
        return base_dir / "models" / "announcement_classifier.joblib"

    def _get_default_data_path(self) -> Path:
        base_dir = Path(__file__).resolve().parent.parent
        return base_dir / "data" / "sample_announcements.csv"

    def _load_or_train(self) -> None:
        """Loads serialized model if present, otherwise trains on default dataset."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                return
            except Exception:
                # If corrupt or version mismatch, retrain
                pass

        data_path = self._get_default_data_path()
        if data_path.exists():
            self.train(str(data_path))
        else:
            # Fallback to rule-based baseline until training data is provided
            self.pipeline = None

    def train(self, csv_path: str) -> Dict[str, float]:
        """
        Trains the scikit-learn pipeline on CSV data.
        Returns training metrics.
        """
        df = pd.read_csv(csv_path)
        if "text" not in df.columns or "category" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'category' columns.")

        X = df["text"].astype(str)
        y = df["category"].astype(str)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=4000,
                sublinear_tf=True,
                strip_accents="unicode",
                stop_words="english",
            )),
            ("clf", LogisticRegression(
                C=2.0,
                max_iter=1000,
                random_state=42,
            )),
        ])

        self.pipeline.fit(X, y)

        # Save model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)

        train_acc = float(self.pipeline.score(X, y))
        return {"train_accuracy": train_acc, "samples_trained": len(df)}

    def rule_based_predict(self, text: str) -> Tuple[str, float, Dict[str, float], List[str]]:
        """
        Transparent rule-based baseline classifier using keyword heuristics.
        Returns: (predicted_category, confidence, probability_distribution, matched_keywords)
        """
        text_lower = text.lower()
        scores: Dict[str, float] = {cat: 0.0 for cat in CATEGORIES}
        matched_by_cat: Dict[str, List[str]] = {cat: [] for cat in CATEGORIES}

        for cat, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                # Exact or phrase boundary match
                pattern = r"\b" + re.escape(kw) + r"\b"
                matches = re.findall(pattern, text_lower)
                if matches:
                    weight = 2.0 if len(kw.split()) > 1 else 1.0
                    scores[cat] += len(matches) * weight
                    matched_by_cat[cat].append(kw)

        total_score = sum(scores.values())
        if total_score == 0:
            # Neutral uniform distribution defaulting to General
            prob_dist = {cat: round(1.0 / len(CATEGORIES), 3) for cat in CATEGORIES}
            return "General", 0.30, prob_dist, []

        prob_dist = {cat: round(scores[cat] / total_score, 3) for cat in CATEGORIES}
        best_cat = max(scores, key=scores.get)
        confidence = prob_dist[best_cat]
        return best_cat, confidence, prob_dist, matched_by_cat[best_cat]

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies an announcement text into one of 7 categories.
        Combines machine learning with transparent keyword matching.
        """
        if not text or not text.strip():
            return {
                "category": "General",
                "confidence": 0.0,
                "probabilities": {cat: 0.0 for cat in CATEGORIES},
                "method": "Error (Empty Input)",
                "matched_keywords": [],
            }

        text_clean = text.strip()
        rule_cat, rule_conf, rule_probs, rule_matches = self.rule_based_predict(text_clean)

        # If ML pipeline is available
        if self.pipeline is not None:
            try:
                probs_array = self.pipeline.predict_proba([text_clean])[0]
                classes = list(self.pipeline.classes_)
                ml_probs = {cls: float(prob) for cls, prob in zip(classes, probs_array)}

                # Fill any missing categories with 0.0
                for cat in CATEGORIES:
                    if cat not in ml_probs:
                        ml_probs[cat] = 0.0

                ml_cat = max(ml_probs, key=ml_probs.get)
                ml_conf = ml_probs[ml_cat]

                # If rule matches strong domain phrases (e.g. "internship", "placement"),
                # combine with ML probability for maximum accuracy
                if rule_matches and ml_cat != rule_cat and rule_conf >= 0.50:
                    # Give weight to strong deterministic keywords
                    combined_probs = {}
                    for cat in CATEGORIES:
                        combined_probs[cat] = 0.65 * ml_probs.get(cat, 0.0) + 0.35 * rule_probs.get(cat, 0.0)
                    final_cat = max(combined_probs, key=combined_probs.get)
                    final_conf = combined_probs[final_cat]
                    method = "Hybrid (ML + Domain Rules)"
                    final_probs = {cat: round(val, 3) for cat, val in combined_probs.items()}
                else:
                    final_cat = ml_cat
                    final_conf = ml_conf
                    method = "Scikit-learn (TF-IDF + Logistic Regression)"
                    final_probs = {cat: round(val, 3) for cat, val in ml_probs.items()}

                # If confidence is extremely low and no domain keywords match (e.g. gibberish or out-of-vocab),
                # default gracefully to 'General' as an out-of-distribution safeguard
                if final_conf < 0.28 and not rule_matches:
                    final_cat = "General"
                    method = "Safeguard Fallback (Out-of-Distribution / Ambiguous)"

                return {
                    "category": final_cat,
                    "confidence": round(float(final_conf), 3),
                    "probabilities": final_probs,
                    "method": method,
                    "matched_keywords": rule_matches,
                }
            except Exception:
                # Graceful fallback to rule-based
                pass

        # If rule matched nothing and ML is not available
        if rule_conf <= 0.30 and not rule_matches:
            rule_cat = "General"

        return {
            "category": rule_cat,
            "confidence": round(float(rule_conf), 3),
            "probabilities": rule_probs,
            "method": "Transparent Rule-Based Baseline",
            "matched_keywords": rule_matches,
        }


# Singleton instance for high-performance reuse
_DEFAULT_CLASSIFIER: Optional[AnnouncementClassifier] = None


def classify_announcement(text: str) -> Dict[str, Any]:
    """
    Convenience function to classify an announcement text.
    """
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = AnnouncementClassifier()
    return _DEFAULT_CLASSIFIER.predict(text)
