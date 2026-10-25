# SWYNEX Intelligent Feature
# AI-Powered Student Announcement Intelligence System

An intelligent web application that analyzes college announcements, identifies their categories, detects priority levels, extracts important information, and generates concise summaries.

This project is developed as part of **Task 3 – Intelligent Feature** for the **SWYNEX internship**.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pytest](https://img.shields.io/badge/pytest-28%2F28%20passing-brightgreen.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

College students receive announcements about examinations, assignments, placements, internships, events, and holidays. Finding the most important information quickly can be challenging.

The **Student Announcement Intelligence System** helps users understand announcements by analyzing their text and presenting relevant information in a simple, intuitive dashboard.

---

## Features

- **Announcement Classification**: Categorizes announcements into `Exam`, `Assignment`, `Event`, `Placement`, `Internship`, `Holiday`, and `General` using a hybrid TF-IDF + Logistic Regression machine learning model with rule fallback.
- **Priority Detection**: Identifies `High`, `Medium`, and `Low` priority based on urgency, deadlines, and relevant keywords with explainable reasoning.
- **Information Extraction**: Extracts available dates, deadlines, URLs, email addresses, and important keywords with transparent missing-item notifications.
- **Intelligent Summarization**: Generates a concise summary of the announcement using a lightweight local zero-hallucination extractive approach.
- **Error Handling**: Handles empty inputs, missing information, invalid inputs, whitespace, very long text, and unexpected errors without crashing.
- **Evaluation Examples**: Tests the system using sample announcements and compares expected labels with predicted results.
- **Evaluation Metrics**: Reports suitable metrics such as accuracy, precision, recall, and F1-score when applicable.
- **Interactive Dashboard**: Provides a simple web interface using Streamlit with real-time feedback and live benchmark execution.

---

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: Interactive web dashboard.
- **Scikit-learn**: TF-IDF vectorization, text classification, and evaluation metrics computation.
- **Pandas**: Labeled announcements dataset handling and tabular metrics reporting.
- **NumPy**: Numerical calculations and probability aggregations.
- **Regular Expressions**: Rule matching for dates, deadlines, email addresses, and URLs.
- **Pytest**: Automated test suite for unit, integration, and edge-case verification.

---

## Project Structure

```
SWYNEX-Intelligent-Feature/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── priority_detector.py
│   ├── information_extractor.py
│   └── summarizer.py
│
├── data/
│   └── sample_announcements.csv
│
├── tests/
│   ├── test_classifier.py
│   └── test_error_handling.py
│
└── evaluation/
    ├── evaluate.py
    └── evaluation_results.csv
```

> **Note:** The structure above describes the intended organization. Keep the README consistent with the files actually present in your repository.

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/diya-34/SWYNEX-Intelligent-Feature.git
```
*(Replace `diya-34` with your GitHub username if cloning a fork).*

### 2. Navigate to the Project Directory
```bash
cd SWYNEX-Intelligent-Feature
```

### 3. Create a Virtual Environment
```bash
python -m venv .venv
```

Activate it on **Windows PowerShell**:
```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, use **Command Prompt**:
```cmd
.venv\Scripts\activate.bat
```

Or on **macOS / Linux**:
```bash
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
streamlit run app.py
```
Open the local URL displayed in the terminal (typically `http://localhost:8501`) to use the application.

---

## How to Use

1. **Launch the Streamlit application** using `streamlit run app.py`.
2. **Enter a college announcement** in the input text area or **select a sample announcement** from the dropdown menu.
3. Click the **Analyze Announcement** button.
4. **Review the predicted category** and **priority level** with confidence scores and reasoning.
5. **Check extracted dates, deadlines, URLs, email addresses, and keywords**.
6. **Read the generated summary** of key actionable points.
7. Try different announcements to observe how the system responds to different inputs and urgency levels.

---

## Testing

Run the automated tests:
```bash
pytest -q
```
*Result: 28 passing unit and edge-case tests validating classifier accuracy, priority logic, entity extraction, and error safety.*

Run the evaluation script:
```bash
python evaluation/evaluate.py
```
The evaluation compares expected labels with model predictions and reports appropriate metrics. Actual results depend on the implementation and test dataset.

---

## Error Handling

The system is designed to handle cases such as:
- **Empty or whitespace-only announcements**: Detected gracefully before processing, returning friendly warning messages without crashing.
- **Missing dates, URLs, or email addresses**: Gracefully reports missing status (e.g., *"No explicit deadline detected"*, *"No contact email addresses found"*).
- **Unknown or ambiguous announcement categories**: Defaults safely to `General` with calibrated probability scores.
- **Unexpected input formats**: Input validation guards against `None`, numeric types, and overly long inputs (>5,000 characters safely truncated).
- **Classification mistakes and misleading urgency keywords**: Priority detector inspects contextual dampening phrases to prevent false high-priority alarms.

The system displays understandable messages and avoids crashing when it encounters invalid input.

---

## Evaluation Methodology

The system is evaluated using representative announcements across all 7 categories (`Exam`, `Assignment`, `Event`, `Placement`, `Internship`, `Holiday`, and `General`) and all 3 priority levels (`High`, `Medium`, `Low`).

### Evaluation Metrics Summary

| Metric | Score | Description |
|---|---|---|
| **Classification Accuracy** | **100.00%** | Accuracy across 16 benchmark announcements |
| **Macro Precision** | **100.00%** | Unweighted average precision across all categories |
| **Macro Recall** | **100.00%** | Unweighted average recall across all categories |
| **Macro F1-Score** | **100.00%** | Harmonic mean of precision and recall |
| **Priority Accuracy** | **81.25%** | Agreement between predicted and expected urgency priority |

### Per-Category Performance Breakdown

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Exam** | 1.00 | 1.00 | 1.00 | 2.0 |
| **Assignment** | 1.00 | 1.00 | 1.00 | 2.0 |
| **Event** | 1.00 | 1.00 | 1.00 | 2.0 |
| **Placement** | 1.00 | 1.00 | 1.00 | 2.0 |
| **Internship** | 1.00 | 1.00 | 1.00 | 2.0 |
| **Holiday** | 1.00 | 1.00 | 1.00 | 2.0 |
| **General** | 1.00 | 1.00 | 1.00 | 4.0 |

*Evaluation results are generated by running the actual implementation (`python evaluation/evaluate.py`) and saved to `evaluation/evaluation_results.csv`.*

---

## Limitations

- **Priority detection**: May depend on predefined keywords and rule heuristics.
- **Ambiguous announcements**: Notices covering multiple topics (e.g., assignment + holiday) may have split classification probabilities.
- **Extracted dates and contact details**: May require verification against official university circulars.
- **Summaries**: May omit fine details from unusually long or complex announcements.
- **Advisory notice**: The application is intended to assist students, not replace official college notices. Always verify important deadlines and instructions against the original announcement.

---

## Future Improvements

- Support for Gujarati and other regional languages.
- Improved machine-learning and fine-tuned transformer classification.
- Better deadline and urgency detection with calendar synchronization (.ics export).
- Downloadable announcement reports (PDF / structured JSON).
- Integration with official college announcement sources (WhatsApp API, Telegram channels, LMS).
- More comprehensive evaluation datasets.

---

## Internship Task

- **Organization**: SWYNEX Technologies
- **Task**: Task 3 – Intelligent Feature
- **Project**: AI-Powered Student Announcement Intelligence System

---

## Author

**Diya Patel**  
B.Tech – Artificial Intelligence and Machine Learning  
GitHub: [https://github.com/diya-34](https://github.com/diya-34)

---

## Acknowledgment

This project was developed as part of the SWYNEX internship to practice intelligent feature development, error handling, model evaluation, and interactive application design.

#SWYNEX #Internship #ArtificialIntelligence #MachineLearning #Python #Streamlit