# AI Job Matching Engine

An AI-powered application that matches candidates with suitable job opportunities based on skills, experience, and profile information.

## Features

- **Job Recommendations:** Find suitable jobs for candidates.
- **Candidate Ranking:** Rank candidates for a selected job using match scores.
- **Skill Gap Analysis:** Identify missing skills required for a job.
- **Match Score Calculation:** Combine text similarity, skill matching, and experience matching.
- **Dashboard:** View candidates, jobs, and match statistics.
- **CSV Export:** Download candidate-job matching results.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity

## Project Structure

```text
AI-Job-Matching-Engine/
├── app.py
├── main.py
├── main_sample_backup.py
├── requirements.txt
├── README.md
├── data/
│   ├── candidates.csv
│   └── jobs.csv
└── tests/
    └── test_engine.py
```

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/aishwaryatechprofile-lab/AI-Job-Matching-Engine.git
   ```

2. Open the project folder.

3. Create and activate a virtual environment if needed.

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the dashboard:

   ```bash
   streamlit run app.py
   ```

## How It Works

The system compares candidate profiles with job descriptions using TF-IDF and cosine similarity. It also calculates skill overlap and experience scores to generate an overall matching score.

## Note

The matching score is an estimate to support job discovery. It does not guarantee suitability or employment.

## Author

Aishwarya M.S.
