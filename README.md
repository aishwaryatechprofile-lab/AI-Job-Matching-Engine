# AI Job Matching Engine

**Task ID:** AI-RH-005  
**Student Code:** DAS011020  
**Technology:** Python, Pandas, NLP preprocessing, TF-IDF, scikit-learn

## Overview
This project ranks jobs for a candidate and ranks candidates for a job. It calculates a weighted score using text similarity, required skills, and experience.

## Features
- Reads job and candidate profiles from CSV files
- Cleans and preprocesses text
- Extracts known skills from profile and job text
- Converts text to TF-IDF vectors
- Uses cosine similarity to compare job and candidate text
- Calculates a weighted score: 60% text similarity + 30% skill match + 10% experience match
- Recommends jobs and ranks candidates
- Reports matched skills and skill gaps
- Exports all pairwise match results to `reports/all_matches.csv`
- Interactive command-line menu

## Project structure
```text
AI_Job_Matching_Engine/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── jobs.csv
│   └── candidates.csv
├── reports/                 # Created when exporting results
└── tests/
    └── test_engine.py
```

## Setup (Windows PowerShell)
Open the terminal in this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

If you already have a `.venv`, activate it and only run `python -m pip install -r requirements.txt`.

## How to use
1. Choose `1` to see available jobs and candidates.
2. Choose `2` and enter a candidate ID such as `C001` to see job recommendations.
3. Choose `3` and enter a job ID such as `J001` to rank candidates.
4. Choose `4` to see matched skills and skills to develop for a candidate-job pair.
5. Choose `5` to export every pair's score to a CSV file.
6. Choose `0` to exit.

## Score calculation
- **Text similarity (60%)**: cosine similarity between TF-IDF vectors for the job and candidate profile.
- **Skill match (30%)**: number of matched required skills divided by the number of required skills.
- **Experience match (10%)**: candidate experience divided by required experience, capped at 100%.

Final score:
`(text_similarity × 0.60 + skill_match × 0.30 + experience_match × 0.10) × 100`

## Notes and limitations
- The included job and candidate records are demonstration data, not live vacancies or real applicants.
- The skill extractor uses a defined vocabulary in `main.py`; add new skills to `SKILL_CATEGORIES` when needed.
- A match score is a ranking aid, not a hiring decision. Recruiters should review candidates fairly and manually.
- No database is required; the project stores data in CSV files.
