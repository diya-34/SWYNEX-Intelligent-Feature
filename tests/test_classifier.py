"""
SWYNEX Intelligent Feature 2 - Automated Test Suite: Classifier & Intelligence Modules
Tests classification accuracy, class probabilities, priority assignment,
information extraction, and zero-hallucination summarization.
"""

import pytest
from src.classifier import classify_announcement, CATEGORIES, AnnouncementClassifier
from src.priority_detector import detect_priority
from src.information_extractor import extract_information
from src.summarizer import summarize_announcement


class TestAnnouncementClassifier:
    """Tests the hybrid classification module."""

    @pytest.fixture(scope="class")
    @classmethod
    def classifier(cls):
        return AnnouncementClassifier()

    def test_supported_categories_count(self):
        assert len(CATEGORIES) == 7
        assert "Exam" in CATEGORIES
        assert "Assignment" in CATEGORIES
        assert "Event" in CATEGORIES
        assert "Placement" in CATEGORIES
        assert "Internship" in CATEGORIES
        assert "Holiday" in CATEGORIES
        assert "General" in CATEGORIES

    @pytest.mark.parametrize("text,expected_category", [
        ("The End-Semester Examination timetable has been released. Admit cards available.", "Exam"),
        ("Submit your Machine Learning Assignment 2 before Friday 11:59 PM on Canvas LMS.", "Assignment"),
        ("Join us for HackSphere 2026, the 24-hour national hackathon in the main auditorium.", "Event"),
        ("Microsoft campus placement drive: Shortlisted candidates technical interviews tomorrow.", "Placement"),
        ("6-month winter internship opportunity with Rs 25,000 monthly stipend at research lab.", "Internship"),
        ("The college will remain closed on Monday for Dussehra holiday. Classes resume Tuesday.", "Holiday"),
        ("Notice: Smart biometric ID cards ready for collection at administrative counter 4.", "General"),
    ])
    def test_category_predictions(self, classifier, text, expected_category):
        res = classifier.predict(text)
        assert res["category"] == expected_category
        assert 0.0 <= res["confidence"] <= 1.0
        assert isinstance(res["probabilities"], dict)
        assert len(res["probabilities"]) == 7

    def test_probabilities_sum_to_one(self, classifier):
        text = "Mid-term examination schedule for Computer Science department."
        res = classifier.predict(text)
        prob_sum = sum(res["probabilities"].values())
        assert pytest.approx(prob_sum, abs=0.05) == 1.0


class TestPriorityDetector:
    """Tests priority scoring and explainable reason generation."""

    def test_high_priority_imminent_deadline(self):
        text = "URGENT: Submit exam registration with late fee strictly tonight before 11:59 PM or face debarment."
        res = detect_priority(text, category="Exam")
        assert res["priority"] == "High"
        assert res["score"] >= 65
        assert "reason" in res and len(res["reason"]) > 10
        assert len(res["urgency_factors"]) > 0

    def test_medium_priority_standard_task(self):
        text = "Database homework assignment posted on LMS. Please submit solutions by next Wednesday."
        res = detect_priority(text, category="Assignment")
        assert res["priority"] in ["Medium", "High"]
        assert "reason" in res and len(res["reason"]) > 10

    def test_low_priority_informational(self):
        text = "Notice: Campus library extended reading room hours will be in effect next week for your information."
        res = detect_priority(text, category="General")
        assert res["priority"] == "Low"
        assert res["score"] < 50
        assert "reason" in res and len(res["reason"]) > 10


class TestInformationExtractor:
    """Tests entity extraction for dates, URLs, emails, and action tags."""

    def test_extract_dates_and_deadlines(self):
        text = "Submit the project report before Friday 5:00 PM on October 25, 2026."
        res = extract_information(text)
        assert len(res["dates"]) > 0
        assert any("October 25" in d or "Friday" in d for d in res["dates"])
        assert len(res["deadlines"]) > 0

    def test_extract_urls_and_emails(self):
        text = "Register at https://hackathon.college.edu or email coordinator at contact@college.ac.in for help."
        res = extract_information(text)
        assert "https://hackathon.college.edu" in res["urls"]
        assert "contact@college.ac.in" in res["emails"]

    def test_extract_action_keywords(self):
        text = "Students must collect their Hall Ticket and Admit Card from ERP Portal."
        res = extract_information(text)
        assert "Hall Ticket" in res["keywords"]
        assert "Admit Card" in res["keywords"]
        assert "ERP Portal" in res["keywords"]

    def test_missing_info_polite_messages(self):
        text = "General discussion on curriculum updates."
        res = extract_information(text)
        assert len(res["urls"]) == 0
        assert "No URLs" in res["status_messages"]["urls"]
        assert len(res["emails"]) == 0
        assert "No contact email" in res["status_messages"]["emails"]


class TestSummarizer:
    """Tests lightweight zero-hallucination extractive summarizer."""

    def test_short_announcement_passthrough(self):
        short_text = "Mid-term exam seating arrangement has been published outside Room 204."
        res = summarize_announcement(short_text)
        assert res["summary"] == short_text
        assert res["sentences_selected"] == 1

    def test_long_announcement_compression(self):
        long_text = (
            "The End-Semester Examination timetable for B.Tech Semester 5 is officially released. "
            "All students must download admit cards before Friday 5:00 PM from the university portal. "
            "Students having clashes must report to the examination office immediately with printed proofs. "
            "Any student found without a stamped hall ticket will not be permitted inside the test hall."
        )
        res = summarize_announcement(long_text, max_sentences=2)
        assert len(res["summary"]) < len(long_text)
        assert res["sentences_selected"] <= 2
        # Zero hallucination check: every selected sentence is a verbatim substring of original
        sentences = [s.strip() for s in res["summary"].split(".") if s.strip()]
        for sent in sentences:
            assert sent in long_text
