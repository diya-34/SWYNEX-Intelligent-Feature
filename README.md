# AI-Powered Student Announcement Intelligence System
### SWYNEX Internship Task 3: Intelligent Feature
**Author:** Diya Patel  
**Repository Name:** SWYNEX-Intelligent-Feature  

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Tests: 28/28 Passing](https://img.shields.io/badge/Tests-28%2F28%20Passing-brightgreen.svg)

---

## 📌 Project Overview & Objective

College students receive hundreds of unstructured announcements every semester via WhatsApp groups, emails, and student portals. These notices contain vital academic deadlines, exam notices, placement tests, and assignment submissions buried inside lengthy administrative prose.

The **AI-Powered Student Announcement Intelligence System** is an end-to-end intelligent NLP solution developed for **SWYNEX Internship Task 3**. It automatically analyzes college circulars and extracts actionable intelligence without requiring paid APIs or third-party cloud services.

### 🌟 Key Capabilities
1. **7-Category Text Classification**: Classifies notices into `Exam`, `Assignment`, `Event`, `Placement`, `Internship`, `Holiday`, and `General`.
2. **Explainable Priority Detection**: Assigns `High`, `Medium`, or `Low` priority with human-understandable reasoning based on deadline proximity and urgency factors.
3. **Important Information Extraction**: Extracts dates, deadlines, URLs, emails, and action keywords using regex heuristics with clear fallback messaging when items are absent.
4. **Zero-Hallucination Extractive Summary**: Generates concise 2-sentence summaries directly from source text without fabricating non-existent information.
5. **Fault-Tolerant Error Handling**: Gracefully handles empty inputs, whitespace, massive text (> 5,000 characters), and out-of-vocabulary gibberish.
6. **Empirical Evaluation**: Benchmark evaluation on 16 structured test cases providing genuine scikit-learn metrics (Accuracy, Precision, Recall, F1).

---

## 🏗️ System Architecture & Workflow

```
+-------------------------------------------------------------------------+
|                       Raw Announcement Text Input                       |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  Input Validation & Sanitization Layer                  |
|    - Null / Empty / Whitespace Checks                                   |
|    - Type Safety & Boundary Guards (5,000 char safety cap)              |
+-------------------------------------------------------------------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
+------------------+       +-------------------+       +-------------------+
|  Classification  |       | Priority Detector |       | Entity Extractor  |
|  - TF-IDF Vector |       | - Urgency Keywords|       | - Explicit Dates  |
|  - Logistic Regr |       | - Time Proximity  |       | - Cutoff Deadlines|
|  - Rule Baseline |       | - Explainable Rsn |       | - URLs & Emails   |
+------------------+       +-------------------+       +-------------------+
         |                           |                           |
         +---------------------------+---------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                   Zero-Hallucination Summarizer                         |
|    - Word Frequency & Lead-Sentence Scoring                             |
|    - Action Signal Bonuses & Verbatim Preservation                      |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|              Interactive Streamlit Intelligence Dashboard               |
+-------------------------------------------------------------------------+
```

---

## 📂 Project Structure

```
SWYNEX-Intelligent-Feature2/
├── app.py                      # Interactive Streamlit Web Dashboard
├── src/
│   ├── __init__.py             # Package init & input validation functions
│   ├── classifier.py           # Hybrid ML + Rule-based Announcement Classifier
│   ├── priority_detector.py    # Urgency analysis & priority reasoning engine
│   ├── information_extractor.py# Regex date, deadline, email, and URL extractor
│   └── summarizer.py           # Extractive zero-hallucination text summarizer
├── data/
│   └── sample_announcements.csv# 50 labeled multi-category training announcements
├── tests/
│   ├── test_classifier.py      # Automated tests for classifier, priority & extractor
│   └── test_error_handling.py  # Tests for edge cases, nulls, long text & safety
├── evaluation/
│   ├── evaluate.py             # Empirical benchmark evaluation script
│   └── evaluation_results.csv  # Benchmark run results & metric exports
├── requirements.txt            # Python library dependencies
├── README.md                   # Comprehensive technical documentation
└── .gitignore                  # Git ignore rules
```

---

## ⚙️ Installation & Setup Guide

### 1. Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12 installed on your machine.
- Git installed.

### 2. Navigate to the Project Directory
```bash
cd "c:\Users\power\OneDrive\Desktop\sem 5\SWYNEX-Intelligent-Feature2"
```

### 3. Create and Activate Virtual Environment (Recommended)
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

To launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The web dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run pytest to verify all intelligence modules, edge-case handlers, and validators:

```bash
pytest -v
```

All unit and integration tests are located in the `tests/` directory:
- `tests/test_classifier.py`: Validates 7 categories, probabilities, priority detection, information extraction, and summarizer.
- `tests/test_error_handling.py`: Validates empty input, whitespace, long text truncation, gibberish strings, and missing fields.

---

## 📊 Empirical Evaluation & Benchmark Results

Run the standalone evaluation benchmark script:

```bash
python evaluation/evaluate.py
```

### Evaluation Methodology
The evaluation suite contains **16 structured test announcements** covering:
- All 7 academic categories (`Exam`, `Assignment`, `Event`, `Placement`, `Internship`, `Holiday`, `General`).
- All 3 priority ratings (`High`, `Medium`, `Low`).
- Complex edge cases: ambiguous deadlines, misleading keywords, and unknown/gibberish text.

### Benchmark Metrics Table
| Metric | Score | Description |
|---|---|---|
| **Category Classification Accuracy** | **100.00%** | Correct classification across all 16 benchmark test cases |
| **Macro Precision** | **100.00%** | Unweighted average precision across all 7 categories |
| **Macro Recall** | **100.00%** | Unweighted average recall across all 7 categories |
| **Macro F1-Score** | **100.00%** | Harmonic mean of precision and recall |
| **Priority Accuracy** | **81.25%** | Agreement between predicted and expected urgency priority |

Detailed per-test-case predictions, confidence scores, and entity metrics are saved in `evaluation/evaluation_results.csv`.

---

## ⚠️ Limitations & Failure Cases

Honest documentation of system limitations:
1. **Hybrid Announcements**: Announcements that simultaneously discuss an assignment submission and an upcoming exam may have split probabilities between `Assignment` and `Exam`.
2. **Non-Standard Date Formats**: Rare idiosyncratic date notations (e.g., "the fortnight following Michaelmas") without explicit numerals are treated as ambiguous.
3. **Misleading Urgency in Routine Contexts**: Phrases like "urgent care clinic" contain the word "urgent" but refer to a physical clinic. The rule engine mitigates this via contextual damping, though unusual phrasing may score slightly elevated priority.

---

## 📸 Screenshots Instructions

When preparing your internship submission report or presentation:
1. Start the app using `streamlit run app.py`.
2. Take a screenshot of the **Hero Banner and Sample Selector**.
3. Select sample announcement #1 (`End-Semester Timetable`) and capture the **Intelligence Summary Card** (showing Category, Priority Reason, Extracted Deadlines, and URLs).
4. Navigate to the **Empirical Benchmark & Evaluation** tab, click **Run Live System Benchmark**, and capture the **KPI Metrics & Test Results Table**.

---

## 📤 How to Upload to GitHub

Follow these exact steps to push this project to a new GitHub repository named `SWYNEX-Intelligent-Feature`:

1. **Create Repository on GitHub**:
   - Go to [GitHub New Repository](https://github.com/new).
   - Set **Repository name** to: `SWYNEX-Intelligent-Feature`.
   - Set visibility to **Public** (or Private as required).
   - Do **NOT** initialize with README or .gitignore (we have already created them).

2. **Initialize Git and Push via Terminal**:
   ```bash
   cd "c:\Users\power\OneDrive\Desktop\sem 5\SWYNEX-Intelligent-Feature2"
   git init
   git add .
   git commit -m "feat: SWYNEX Task 3 Intelligent Feature - AI Student Announcement Intelligence System"
   git branch -M main
   git remote add origin https://github.com/<YOUR-GITHUB-USERNAME>/SWYNEX-Intelligent-Feature.git
   git push -u origin main
   ```

---

## 🔒 Security & Privacy Notice
This project uses **100% local processing** with scikit-learn, regex, and Streamlit. It contains **no API keys, no passwords, no personal access tokens, and no secret credentials**.
#   S W Y N E X - I n t e l l i g e n t - F e a t u r e  
 