"""
SWYNEX Intelligent Feature 2 - System Evaluation & Benchmark Script
Runs rigorous empirical evaluation across 16 curated test announcements,
computes scikit-learn classification metrics (Accuracy, Precision, Recall, F1),
tests edge/failure cases, and exports results to evaluation_results.csv.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add project root to path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import classify_announcement, CATEGORIES
from src.priority_detector import detect_priority
from src.information_extractor import extract_information
from src.summarizer import summarize_announcement
from src import validate_input


# 16 Curated Evaluation Test Cases covering all 7 categories, 3 priority levels, and edge cases
EVALUATION_DATASET: List[Dict[str, Any]] = [
    {
        "id": "TC-01",
        "description": "Standard Exam with Imminent Deadline",
        "text": "The End-Semester Examination timetable for B.Tech Semester 5 is officially released. All students must download admit cards before Friday 5:00 PM at https://exam.university.edu.",
        "expected_category": "Exam",
        "expected_priority": "High",
        "has_deadline": True,
        "is_edge_case": False,
    },
    {
        "id": "TC-02",
        "description": "Exam Schedule Clash Notice",
        "text": "Notice regarding exam timetable clash: Students having two examination papers on the same day must report to the Controller of Examinations before 4:00 PM tomorrow with printed hall tickets.",
        "expected_category": "Exam",
        "expected_priority": "High",
        "has_deadline": True,
        "is_edge_case": False,
    },
    {
        "id": "TC-03",
        "description": "Coding Assignment Submission",
        "text": "Submission of Machine Learning Assignment 3 on Neural Networks is due tomorrow by 11:59 PM on Google Classroom. Late submissions will receive zero marks.",
        "expected_category": "Assignment",
        "expected_priority": "High",
        "has_deadline": True,
        "is_edge_case": False,
    },
    {
        "id": "TC-04",
        "description": "Long-Term Project Assignment",
        "text": "Database Management Systems homework assignment on SQL query optimization has been posted. Please submit your solutions by November 15 to turnitin@college.ac.in.",
        "expected_category": "Assignment",
        "expected_priority": "Medium",
        "has_deadline": True,
        "is_edge_case": False,
    },
    {
        "id": "TC-05",
        "description": "Inter-College Hackathon Event",
        "text": "Annual Inter-College Hackathon 'HackSphere 2026' registration is now open! Win cash prizes up to Rs 1,00,000. Register at https://hacksphere.dev before November 5.",
        "expected_category": "Event",
        "expected_priority": "Medium",
        "has_deadline": True,
        "is_edge_case": False,
    },
    {
        "id": "TC-06",
        "description": "Academic Guest Lecture Event",
        "text": "The Department of Computer Science is hosting a guest lecture on 'Generative AI and Large Language Models' on Friday at 3:00 PM in the Main Auditorium. All welcome.",
        "expected_category": "Event",
        "expected_priority": "Low",
        "has_deadline": False,
        "is_edge_case": False,
    },
    {
        "id": "TC-07",
        "description": "Top Tier Campus Placement Drive",
        "text": "Microsoft Campus Placement Drive 2026: Eligible B.Tech 7th semester students with CGPA >= 8.0 must register on the CDC portal before tomorrow 12:00 PM.",
        "expected_category": "Placement",
        "expected_priority": "High",
        "has_deadline": True,
        "is_edge_case": False,
    },
    {
        "id": "TC-08",
        "description": "Campus Placement Training",
        "text": "Placement training session on Resume Building and Mock Technical Interviews will be held tomorrow at 3:00 PM in Seminar Hall 2.",
        "expected_category": "Placement",
        "expected_priority": "Medium",
        "has_deadline": False,
        "is_edge_case": False,
    },
    {
        "id": "TC-09",
        "description": "Industrial Stipend Internship",
        "text": "Winter Internship Opportunity at Tech Mahindra: 6-month industrial internship with a stipend of Rs 25,000/month. Apply before November 10 at intern@techmahindra.com.",
        "expected_category": "Internship",
        "expected_priority": "High",
        "has_deadline": True,
        "is_edge_case": False,
    },
    {
        "id": "TC-10",
        "description": "Research Internship Call",
        "text": "Summer Research Internship 2027 at Indian Institute of Science (IISc): Applications open for 3rd year students. Apply before December 15 with recommendation letters.",
        "expected_category": "Internship",
        "expected_priority": "Medium",
        "has_deadline": True,
        "is_edge_case": False,
    },
    {
        "id": "TC-11",
        "description": "Festival Holiday Circular",
        "text": "College will remain closed on Monday, October 26 on account of Dussehra festival. All classes, labs, and office operations will be suspended.",
        "expected_category": "Holiday",
        "expected_priority": "Medium",
        "has_deadline": False,
        "is_edge_case": False,
    },
    {
        "id": "TC-12",
        "description": "Emergency Weather Closure",
        "text": "Emergency Closure: Due to severe cyclone alert, the campus will be evacuated today by 2:00 PM. All evening laboratories and sports activities are cancelled.",
        "expected_category": "Holiday",
        "expected_priority": "High",
        "has_deadline": True,
        "is_edge_case": False,
    },
    {
        "id": "TC-13",
        "description": "Routine Administrative Notice",
        "text": "All students must collect their renewed biometric smart ID cards from the administrative office counter 4 between 10:00 AM and 1:00 PM on working days.",
        "expected_category": "General",
        "expected_priority": "Low",
        "has_deadline": False,
        "is_edge_case": False,
    },
    {
        "id": "TC-14",
        "description": "Edge Case: Missing Deadline / Ambiguous Timing",
        "text": "General guidelines for upcoming open elective courses will be published on the institute portal in due course. Students are advised to check back soon.",
        "expected_category": "General",
        "expected_priority": "Low",
        "has_deadline": False,
        "is_edge_case": True,
    },
    {
        "id": "TC-15",
        "description": "Edge Case: Misleading Urgency Keyword in Routine Context",
        "text": "The urgent care first-aid health clinic is located on the ground floor next to the student cafeteria. Regular consultation hours are 9:00 AM to 5:00 PM.",
        "expected_category": "General",
        "expected_priority": "Low",
        "has_deadline": False,
        "is_edge_case": True,
    },
    {
        "id": "TC-16",
        "description": "Edge Case: Unknown Gibberish Text",
        "text": "asdfghjk qwertyuiop zxcvbnm kljhgfds aoiweury 1234567890 testing out of vocabulary input.",
        "expected_category": "General",
        "expected_priority": "Low",
        "has_deadline": False,
        "is_edge_case": True,
    },
]


def run_evaluation() -> Dict[str, Any]:
    """
    Executes benchmark evaluation across all test cases.
    Returns calculated metrics and detailed results dataframe.
    """
    results: List[Dict[str, Any]] = []
    y_true_cat: List[str] = []
    y_pred_cat: List[str] = []

    for item in EVALUATION_DATASET:
        text = item["text"]
        is_valid, cleaned, error, warning = validate_input(text)

        # Run intelligent pipeline
        clf_res = classify_announcement(cleaned)
        pred_cat = clf_res["category"]
        conf = clf_res["confidence"]

        extracted = extract_information(cleaned)
        prio_res = detect_priority(cleaned, category=pred_cat)
        pred_prio = prio_res["priority"]

        summary_res = summarize_announcement(cleaned)

        cat_correct = (pred_cat == item["expected_category"])
        prio_correct = (pred_prio == item["expected_priority"])

        y_true_cat.append(item["expected_category"])
        y_pred_cat.append(pred_cat)

        results.append({
            "test_id": item["id"],
            "description": item["description"],
            "text": text[:60] + "..." if len(text) > 60 else text,
            "expected_category": item["expected_category"],
            "predicted_category": pred_cat,
            "category_match": cat_correct,
            "confidence": conf,
            "expected_priority": item["expected_priority"],
            "predicted_priority": pred_prio,
            "priority_match": prio_correct,
            "priority_score": prio_res["score"],
            "deadlines_found": len(extracted["deadlines"]),
            "urls_found": len(extracted["urls"]),
            "emails_found": len(extracted["emails"]),
            "summary_len": len(summary_res["summary"]),
            "is_edge_case": item["is_edge_case"],
        })

    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = Path(__file__).resolve().parent / "evaluation_results.csv"
    results_df.to_csv(output_path, index=False)

    # Compute empirical metrics
    accuracy = accuracy_score(y_true_cat, y_pred_cat)
    clf_rep = classification_report(
        y_true_cat,
        y_pred_cat,
        labels=CATEGORIES,
        output_dict=True,
        zero_division=0,
    )

    priority_accuracy = sum(r["priority_match"] for r in results) / len(results)

    return {
        "results_df": results_df,
        "total_cases": len(results),
        "accuracy": round(float(accuracy), 4),
        "priority_accuracy": round(float(priority_accuracy), 4),
        "macro_f1": round(float(clf_rep["macro avg"]["f1-score"]), 4),
        "macro_precision": round(float(clf_rep["macro avg"]["precision"]), 4),
        "macro_recall": round(float(clf_rep["macro avg"]["recall"]), 4),
        "classification_report": clf_rep,
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("SWYNEX Task 3 - Intelligent Feature: Empirical Benchmark Evaluation")
    print("=" * 70)
    metrics = run_evaluation()
    print(f"\nEvaluation Results across {metrics['total_cases']} benchmark cases:")
    print(f"- Category Classification Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"- Macro Precision:                {metrics['macro_precision'] * 100:.2f}%")
    print(f"- Macro Recall:                   {metrics['macro_recall'] * 100:.2f}%")
    print(f"- Macro F1-Score:                 {metrics['macro_f1'] * 100:.2f}%")
    print(f"- Priority Accuracy:              {metrics['priority_accuracy'] * 100:.2f}%")
    print(f"\nDetailed results saved to: {metrics['output_path']}\n")
    print("Per-Category Metrics:")
    for cat in CATEGORIES:
        if cat in metrics["classification_report"]:
            c_data = metrics["classification_report"][cat]
            print(f"  [{cat:12s}] Precision: {c_data['precision']:.2f} | Recall: {c_data['recall']:.2f} | F1: {c_data['f1-score']:.2f} (Support: {c_data['support']})")
    print("=" * 70)
