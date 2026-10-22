"""
SWYNEX Intelligent Feature 2 - AI-Powered Student Announcement Intelligence System
Interactive Streamlit Application
Author: Diya Patel (SWYNEX Internship Task 3)

Features:
- Live Announcement Analysis: Classification, Priority Scoring, Entity Extraction & Summarization
- One-Click Sample Announcement Selector
- Empirical Benchmark Evaluation Dashboard (Accuracy, F1, Precision, Recall, Confusion Report)
- Edge-Case & Failure Analysis
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np

# Ensure src modules are resolvable
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src import (
    classify_announcement,
    detect_priority,
    extract_information,
    summarize_announcement,
    validate_input,
    CATEGORIES,
)
from evaluation.evaluate import EVALUATION_DATASET, run_evaluation


# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SWYNEX - Announcement Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        border-radius: 16px;
        padding: 2rem 2.2rem;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        color: #c7d2fe;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        border: 1px solid rgba(165, 180, 252, 0.3);
        margin-bottom: 0.75rem;
        letter-spacing: 0.5px;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-top: 0.5rem;
        line-height: 1.5;
        max-width: 900px;
    }

    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }

    .tag-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    .tag-exam { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .tag-assignment { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .tag-event { background: #e0e7ff; color: #3730a3; border: 1px solid #c7d2fe; }
    .tag-placement { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .tag-internship { background: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }
    .tag-holiday { background: #ffedd5; color: #9a3412; border: 1px solid #fed7aa; }
    .tag-general { background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0; }

    .priority-high { background: #fef2f2; color: #dc2626; border-left: 4px solid #dc2626; padding: 0.75rem; border-radius: 6px; }
    .priority-medium { background: #fffbeb; color: #d97706; border-left: 4px solid #d97706; padding: 0.75rem; border-radius: 6px; }
    .priority-low { background: #f0fdf4; color: #16a34a; border-left: 4px solid #16a34a; padding: 0.75rem; border-radius: 6px; }

    .summary-box {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border-left: 4px solid #6366f1;
        font-size: 0.95rem;
        color: #1e293b;
        line-height: 1.6;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sample Announcements for Quick Testing
# -----------------------------------------------------------------------------
SAMPLE_OPTIONS = {
    "Select an example announcement to test...": "",
    "1. [Exam] End-Semester Timetable & Admit Cards (High Priority)": (
        "The End-Semester Examination timetable for B.Tech Semester 5 is officially released. "
        "All students must download admit cards before Friday 5:00 PM at https://exam.university.edu. "
        "Hall ticket is strictly mandatory for entry into examination halls."
    ),
    "2. [Assignment] Machine Learning Project Due Tomorrow (High Priority)": (
        "Submission of Machine Learning Assignment 3 on Neural Networks is due tomorrow by 11:59 PM "
        "on Google Classroom. Late submissions will receive zero marks without exception. "
        "Submit questions to ta-support@college.edu."
    ),
    "3. [Placement] Microsoft Campus Recruitment Drive (High Priority)": (
        "Microsoft Campus Placement Drive 2026: Eligible B.Tech 7th semester students with CGPA >= 8.0 "
        "must register on the CDC portal before tomorrow 12:00 PM. Technical interviews begin Wednesday at 9:00 AM."
    ),
    "4. [Internship] Tech Mahindra Winter Internship with Stipend (High Priority)": (
        "Winter Internship Opportunity at Tech Mahindra: 6-month industrial internship with a stipend "
        "of Rs 25,000/month. Apply before November 10 at intern@techmahindra.com. Open to 3rd & 4th year students."
    ),
    "5. [Event] National HackSphere Hackathon (Medium Priority)": (
        "Annual Inter-College Hackathon 'HackSphere 2026' registration is now open! "
        "Win cash prizes up to Rs 1,00,000. Register at https://hacksphere.dev before November 5."
    ),
    "6. [Holiday] Emergency Cyclone Closure (High Priority)": (
        "Emergency Closure: Due to severe cyclone alert, the campus will be evacuated today by 2:00 PM. "
        "All evening laboratories and sports activities are cancelled until further notice."
    ),
    "7. [Holiday] Dussehra Festival Holiday Notice (Medium Priority)": (
        "College will remain closed on Monday, October 26 on account of Dussehra festival. "
        "All classes, labs, and office operations will be suspended. Regular sessions resume Tuesday."
    ),
    "8. [General] Renewed Smart ID Card Collection (Low Priority)": (
        "All students must collect their renewed biometric smart ID cards from the administrative office "
        "counter 4 between 10:00 AM and 1:00 PM on working days."
    ),
    "9. [Edge Case] Ambiguous Timing / Missing Deadline": (
        "General guidelines for upcoming open elective courses will be published on the institute portal "
        "in due course. Students are advised to check back soon."
    ),
    "10. [Edge Case] Misleading Urgency Keyword in Routine Context": (
        "The campus urgent care first-aid health clinic is located on the ground floor next to the student cafeteria. "
        "Regular consultation hours are 9:00 AM to 5:00 PM."
    ),
}


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Information
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("System Controls")
    st.markdown("**SWYNEX Task 3**\nAI Student Announcement System")

    st.markdown("---")
    st.subheader("Quick Test Samples")
    selected_sample_key = st.selectbox(
        "Choose a pre-configured sample announcement:",
        options=list(SAMPLE_OPTIONS.keys()),
        index=0,
    )
    prefilled_text = SAMPLE_OPTIONS[selected_sample_key]

    st.markdown("---")
    st.subheader("Model Configuration")
    st.write("**Model Type:** TF-IDF + Logistic Regression")
    st.write("**Classes:** 7 Academic Categories")
    st.write("**Priority Engine:** Auditable Rules + Temporal Heuristics")
    st.write("**Summarizer:** Zero-Hallucination Extractive")
    st.caption("Runs 100% locally with Scikit-learn. No external API keys required.")


# -----------------------------------------------------------------------------
# Main Hero Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">SWYNEX INTERNSHIP • TASK 3</div>
    <h1 class="hero-title">AI-Powered Student Announcement Intelligence System</h1>
    <p class="hero-subtitle">
        Automatically categorizes unstructured college announcements, assesses deadline urgency & priority,
        extracts dates, deadlines, URLs, and emails, and produces zero-hallucination summaries.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Live Announcement Analyzer",
    "📊 Empirical Benchmark & Evaluation",
    "📚 Dataset & System Documentation"
])


# =============================================================================
# TAB 1: Live Announcement Analyzer
# =============================================================================
with tab1:
    st.subheader("Analyze College Announcement")

    # Announcement Text Input Area
    announcement_input = st.text_area(
        label="Enter or paste announcement text below:",
        value=prefilled_text,
        height=140,
        placeholder="e.g., The End-Semester Examination timetable for B.Tech Semester 5 is released. Download admit cards before Friday 5:00 PM at https://exam.university.edu...",
        help="Paste any college notice, circular, or WhatsApp announcement."
    )

    col_btn1, col_btn2, col_stats = st.columns([1.5, 1, 4])
    with col_btn1:
        analyze_clicked = st.button("🚀 Analyze Announcement", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🔄 Clear Input", use_container_width=True)
    with col_stats:
        char_count = len(announcement_input)
        word_count = len(announcement_input.split())
        st.caption(f"**Length:** {char_count} characters | {word_count} words")

    if clear_clicked:
        st.rerun()

    # Process and display analysis results
    if analyze_clicked or (prefilled_text and announcement_input == prefilled_text):
        is_valid, cleaned_text, error_msg, warning_msg = validate_input(announcement_input)

        if not is_valid:
            st.error(f"⚠️ Validation Error: {error_msg}")
        else:
            if warning_msg:
                st.warning(f"ℹ️ {warning_msg}")

            # Run intelligent analysis modules
            with st.spinner("Analyzing text with machine learning & entity extractors..."):
                classification = classify_announcement(cleaned_text)
                extracted = extract_information(cleaned_text)
                priority = detect_priority(cleaned_text, category=classification["category"], extracted_dates=extracted["dates"])
                summary = summarize_announcement(cleaned_text, max_sentences=2)

            st.markdown("---")
            st.markdown("### 📋 Intelligence Analysis Summary")

            # Top Cards Row: Category & Priority
            col_cat, col_prio = st.columns(2)

            with col_cat:
                cat_name = classification["category"]
                conf_pct = classification["confidence"] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <span style="font-size:0.85rem; color:#64748b; font-weight:600;">PREDICTED CATEGORY</span>
                    <div style="font-size:1.8rem; font-weight:800; color:#0f172a; margin:0.3rem 0;">
                        {cat_name}
                    </div>
                    <div style="font-size:0.9rem; color:#475569;">
                        Confidence: <b>{conf_pct:.1f}%</b> • Method: <i>{classification['method']}</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(classification["confidence"]))

                # Show matched keywords if available
                if classification.get("matched_keywords"):
                    st.caption(f"**Keywords Detected:** {', '.join(classification['matched_keywords'])}")

            with col_prio:
                prio_name = priority["priority"]
                prio_score = priority["score"]
                css_class = f"priority-{prio_name.lower()}"

                st.markdown(f"""
                <div class="metric-card">
                    <span style="font-size:0.85rem; color:#64748b; font-weight:600;">ASSIGNED PRIORITY</span>
                    <div style="font-size:1.8rem; font-weight:800; margin:0.3rem 0;">
                        <span class="{css_class}">{prio_name.upper()} ({prio_score}/100)</span>
                    </div>
                    <div style="font-size:0.88rem; color:#334155; margin-top:0.4rem;">
                        <b>Reason:</b> {priority['reason']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Middle Row: Extracted Entities
            st.markdown("#### 📌 Extracted Key Information")
            e_col1, e_col2, e_col3, e_col4 = st.columns(4)

            with e_col1:
                st.markdown("**🗓️ Dates & Deadlines**")
                if extracted["deadlines"]:
                    for d in extracted["deadlines"]:
                        st.markdown(f"- 🔴 **Deadline:** `{d}`")
                elif extracted["dates"]:
                    for d in extracted["dates"]:
                        st.markdown(f"- ⚪ Date: `{d}`")
                else:
                    st.info(extracted["status_messages"]["dates"])

            with e_col2:
                st.markdown("**🔗 Portals & URLs**")
                if extracted["urls"]:
                    for u in extracted["urls"]:
                        st.markdown(f"- [{u}]({u})")
                else:
                    st.info(extracted["status_messages"]["urls"])

            with e_col3:
                st.markdown("**✉️ Contact Emails**")
                if extracted["emails"]:
                    for em in extracted["emails"]:
                        st.markdown(f"- [{em}](mailto:{em})")
                else:
                    st.info(extracted["status_messages"]["emails"])

            with e_col4:
                st.markdown("**🏷️ Action Tags**")
                if extracted["keywords"]:
                    for kw in extracted["keywords"]:
                        st.markdown(f"`{kw}` ")
                else:
                    st.info(extracted["status_messages"]["keywords"])

            # Bottom Row: Summary & Recommended Action
            st.markdown("<br>", unsafe_allow_html=True)
            col_sum, col_act = st.columns([1.5, 1])

            with col_sum:
                st.markdown("#### 📝 Intelligent Concise Summary (Zero-Hallucination)")
                st.markdown(f"""
                <div class="summary-box">
                    "{summary['summary']}"
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Original: {summary['original_chars']} chars → Summary: {summary['summary_chars']} chars (Compression: {summary['compression_ratio']})")

            with col_act:
                st.markdown("#### ✅ Student Recommended Next Steps")
                st.info(priority["recommended_action"])

            # Probability Distribution Expander
            with st.expander("📊 View Complete Category Probability Distribution"):
                probs_df = pd.DataFrame(
                    list(classification["probabilities"].items()),
                    columns=["Category", "Probability"]
                ).sort_values("Probability", ascending=False)
                st.dataframe(probs_df.style.format({"Probability": "{:.3f}"}), use_container_width=True)


# =============================================================================
# TAB 2: Empirical Benchmark & Evaluation
# =============================================================================
with tab2:
    st.subheader("System Evaluation & Benchmark Metrics")
    st.markdown("""
    Evaluation is conducted across a structured benchmark suite of **16 realistic announcements**
    covering all 7 categories (`Exam`, `Assignment`, `Event`, `Placement`, `Internship`, `Holiday`, `General`),
    all 3 priority levels (`High`, `Medium`, `Low`), and complex edge cases.
    """)

    if st.button("⚡ Run Live System Benchmark", type="primary"):
        with st.spinner("Executing empirical evaluation across test suite..."):
            metrics = run_evaluation()
            st.session_state["benchmark_metrics"] = metrics

    if "benchmark_metrics" in st.session_state:
        m = st.session_state["benchmark_metrics"]

        # Metric KPI cards
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Accuracy", f"{m['accuracy'] * 100:.1f}%")
        kpi2.metric("Macro Precision", f"{m['macro_precision'] * 100:.1f}%")
        kpi3.metric("Macro Recall", f"{m['macro_recall'] * 100:.1f}%")
        kpi4.metric("Macro F1-Score", f"{m['macro_f1'] * 100:.1f}%")
        kpi5.metric("Priority Match", f"{m['priority_accuracy'] * 100:.1f}%")

        st.markdown("---")
        st.markdown("#### Detailed Test Results Log")
        st.dataframe(
            m["results_df"][["test_id", "description", "expected_category", "predicted_category", "category_match", "confidence", "expected_priority", "predicted_priority", "priority_match", "deadlines_found"]],
            use_container_width=True
        )

        st.markdown("#### Per-Category Performance Breakdown")
        cat_stats = []
        for cat in CATEGORIES:
            if cat in m["classification_report"]:
                rep = m["classification_report"][cat]
                cat_stats.append({
                    "Category": cat,
                    "Precision": f"{rep['precision']:.2f}",
                    "Recall": f"{rep['recall']:.2f}",
                    "F1-Score": f"{rep['f1-score']:.2f}",
                    "Support": rep['support']
                })
        st.table(pd.DataFrame(cat_stats))
    else:
        st.info("Click 'Run Live System Benchmark' above to execute tests and view real evaluation metrics.")


# =============================================================================
# TAB 3: Dataset & System Documentation
# =============================================================================
with tab3:
    st.subheader("System Architecture & Dataset")

    col_doc1, col_doc2 = st.columns(2)

    with col_doc1:
        st.markdown("""
        ### 🏗️ Architecture Pipeline
        1. **Input Validation**: Rejects empty/whitespace input; caps text at 5,000 characters safely.
        2. **Classifier (`classifier.py`)**:
           - Scikit-learn Pipeline with `TfidfVectorizer` (1-2 ngrams, sublinear TF) + `LogisticRegression`.
           - Transparent keyword baseline map for cold-start and explainability.
        3. **Priority Detector (`priority_detector.py`)**:
           - Analyzes urgency keywords (`strictly tonight`, `urgent`, `mandatory`), category stakes, and temporal proximity.
           - Generates human-understandable explanation reasons.
        4. **Entity Extractor (`information_extractor.py`)**:
           - Extracts explicit dates, relative dates, deadlines, URLs, emails, and action tags using regex.
           - Emits friendly polite messages when entities are missing.
        5. **Summarizer (`summarizer.py`)**:
           - Extractive sentence scoring with frequency weighting, lead bias, and action verb boost.
           - **Zero hallucination guarantee**: extracts verbatim sentences without inventing facts.
        """)

    with col_doc2:
        st.markdown("""
        ### 📂 Dataset Overview
        - **File:** `data/sample_announcements.csv`
        - **Total Training Samples:** 50 realistic college announcements
        - **Categories:**
          - 📘 `Exam`
          - 📝 `Assignment`
          - 🎪 `Event`
          - 💼 `Placement`
          - 🔬 `Internship`
          - 🏖️ `Holiday`
          - 📢 `General`
        - **Evaluation Benchmark:** 16 independent test cases (`evaluation/evaluate.py`)
        """)

    st.markdown("---")
    st.caption("Developed for SWYNEX Internship Task 3 — Intelligent Feature. All rights reserved.")
