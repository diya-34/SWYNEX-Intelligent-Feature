# SWYNEX Intelligent Feature

## AI-Powered Student Announcement Intelligence System

A Python-based intelligent web application that analyzes college announcements, identifies categories, detects priority levels, extracts important information, and generates concise summaries.

Developed as part of **Task 3 – Intelligent Feature** for the SWYNEX internship.

---

## Features

- **Announcement Classification:** Classifies announcements into Exam, Assignment, Event, Placement, Internship, Holiday, and General.
- **Priority Detection:** Assigns High, Medium, or Low priority based on urgency and deadlines.
- **Information Extraction:** Extracts dates, deadlines, URLs, email addresses, and keywords.
- **Intelligent Summarization:** Generates concise summaries using a lightweight local approach.
- **Error Handling:** Handles empty input, missing information, and invalid inputs.
- **Evaluation:** Tests predictions against sample announcements and calculates appropriate evaluation metrics.
- **Interactive Dashboard:** Provides a simple user interface using Streamlit.

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Regular Expressions
- Pytest

## Project Structure

```text
SWYNEX-Intelligent-Feature/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── priority_detector.py
│   ├── information_extractor.py
│   └── summarizer.py
├── data/
│   └── sample_announcements.csv
├── tests/
│   ├── test_classifier.py
│   └── test_error_handling.py
└── evaluation/
    ├── evaluate.py
    └── evaluation_results.csv
```

*This structure should match the files actually present in the repository.*

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/diya-34/SWYNEX-Intelligent-Feature.git
```

### 2. Open the Project Folder

```bash
cd SWYNEX-Intelligent-Feature
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Run Tests

```bash
pytest -q
```

## Run Evaluation

```bash
python evaluation/evaluate.py
```

Use the generated evaluation results to review classification performance and identify incorrect predictions. Report only metrics produced by the actual implementation.

## How to Use

1. Launch the Streamlit application.
2. Enter a college announcement or choose a sample.
3. Click **Analyze Announcement**.
4. Review the predicted category and priority.
5. Check extracted dates, URLs, emails, and keywords.
6. Read the generated summary.
7. Test additional announcements and review evaluation results.

## Error Handling

The application should handle:

- Empty or whitespace-only text.
- Missing dates and contact details.
- Unknown or ambiguous announcements.
- Invalid input formats.
- Unexpected processing errors.

## Limitations

- Ambiguous announcements may be classified incorrectly.
- Priority detection may depend on predefined rules and keywords.
- Extracted information should be verified against the original announcement.
- Evaluation metrics depend on the quality and size of the test dataset.

## Future Improvements

- Gujarati-language announcement support.
- Improved machine-learning classification.
- More accurate deadline detection.
- Downloadable reports.
- Expanded evaluation datasets.

## Internship Information

**Organization:** SWYNEX Technologies  
**Task:** Task 3 – Intelligent Feature  
**Project:** Student Announcement Intelligence System

## Author

**Diya Patel**  
B.Tech – Artificial Intelligence and Machine Learning

GitHub: [diya-34](https://github.com/diya-34)

---

#SWYNEX #Internship #ArtificialIntelligence #MachineLearning #Python #Streamlit
