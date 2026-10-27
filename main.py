"""
AI Job Matching Engine
Task ID: AI-RH-005 | Student Code: DAS011020

Matches candidate profiles to job descriptions using:
- Text preprocessing and skill extraction
- TF-IDF vectorization and cosine similarity
- Weighted score: 60% text similarity, 30% skill match, 10% experience match
- Job recommendations, candidate ranking, and skill-gap analysis
"""

from pathlib import Path
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.stem import WordNetLemmatizer
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    LEMMATIZER = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SKILL_CATEGORIES = {
    "programming": [
        "python", "java", "javascript", "c++", "sql", "r"
    ],
    "web": [
        "html", "css", "react", "angular", "vue", "django", "flask",
        "node.js", "rest apis"
    ],
    "data_science": [
        "machine learning", "deep learning", "nlp", "computer vision",
        "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn",
        "statistics", "data visualization", "excel"
    ],
    "database": ["mysql", "postgresql", "mongodb", "oracle", "redis"],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "linux",
        "ci/cd"
    ],
    "cybersecurity": [
        "network security", "cryptography", "penetration testing"
    ],
    "business": [
        "product strategy", "market research", "agile", "communication",
        "data analysis", "teamwork", "leadership", "problem solving"
    ],
}


def preprocess_text(text):
    """Normalize text before vectorization."""
    text = str(text).lower()
    # Keep letters, digits, plus signs and dots for skills such as C++ and Node.js.
    text = re.sub(r"[^a-z0-9+#./\s-]", " ", text)
    tokens = re.findall(r"[a-z0-9+#./-]+", text)
    cleaned = []
    for token in tokens:
        if len(token) <= 1:
            continue
        if LEMMATIZER is not None and token.isalpha():
            try:
                token = LEMMATIZER.lemmatize(token)
            except LookupError:
                pass
        cleaned.append(token)
    return " ".join(cleaned)


def extract_skills(text):
    """Return recognized skills found in a job description or candidate profile."""
    text_lower = str(text).lower()
    found = []
    for skills in SKILL_CATEGORIES.values():
        for skill in skills:
            # Phrase-aware matching prevents 'r' matching any letter inside words.
            if skill == "r":
                if re.search(r"(?<![a-z])r(?![a-z])", text_lower):
                    found.append(skill)
            elif skill in text_lower:
                found.append(skill)
    return sorted(set(found))


class JobMatchingEngine:
    def __init__(self, jobs_path=None, candidates_path=None):
        self.jobs_path = Path(jobs_path) if jobs_path else DATA_DIR / "jobs.csv"
        self.candidates_path = (
            Path(candidates_path) if candidates_path else DATA_DIR / "candidates.csv"
        )
        self.jobs_df = self._load_jobs()
        self.candidates_df = self._load_candidates()
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
        self.job_vectors = None
        self.candidate_vectors = None
        self._create_vectors()

    def _load_jobs(self):
        """Load and validate job records from CSV."""
        df = pd.read_csv(self.jobs_path)
        required = {
            "job_id", "title", "company", "description", "required_skills",
            "experience_required", "location", "salary_range"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"jobs.csv is missing columns: {sorted(missing)}")
        df = df.fillna("")
        df["required_skills_list"] = df["required_skills"].apply(
            lambda value: [s.strip().lower() for s in str(value).split("|") if s.strip()]
        )
        df["combined_text"] = (
            df["title"].astype(str) + " " + df["description"].astype(str) + " " +
            df["required_skills"].astype(str)
        )
        df["clean_text"] = df["combined_text"].apply(preprocess_text)
        return df

    def _load_candidates(self):
        """Load and validate candidate records from CSV."""
        df = pd.read_csv(self.candidates_path)
        required = {
            "candidate_id", "name", "skills", "experience_years", "education",
            "preferred_location", "expected_salary", "profile_summary"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"candidates.csv is missing columns: {sorted(missing)}")
        df = df.fillna("")
        df["skills_list"] = df["skills"].apply(
            lambda value: [s.strip().lower() for s in str(value).split("|") if s.strip()]
        )
        df["combined_text"] = (
            df["skills"].astype(str) + " " + df["education"].astype(str) + " " +
            df["profile_summary"].astype(str)
        )
        df["clean_text"] = df["combined_text"].apply(preprocess_text)
        df["experience_years"] = pd.to_numeric(
            df["experience_years"], errors="coerce"
        ).fillna(0)
        return df

    def _create_vectors(self):
        """Fit TF-IDF once using both datasets, then create vectors."""
        all_text = self.jobs_df["clean_text"].tolist() + self.candidates_df["clean_text"].tolist()
        if not any(text.strip() for text in all_text):
            raise ValueError("Job and candidate text data cannot all be empty.")
        self.vectorizer.fit(all_text)
        self.job_vectors = self.vectorizer.transform(self.jobs_df["clean_text"])
        self.candidate_vectors = self.vectorizer.transform(self.candidates_df["clean_text"])

    def calculate_match_score(self, job_idx, candidate_idx):
        """Calculate the weighted match score for one job-candidate pair."""
        job = self.jobs_df.iloc[job_idx]
        candidate = self.candidates_df.iloc[candidate_idx]

        similarity = float(cosine_similarity(
            self.job_vectors[job_idx:job_idx + 1],
            self.candidate_vectors[candidate_idx:candidate_idx + 1]
        )[0][0])

        required_skills = set(job["required_skills_list"])
        candidate_skills = set(candidate["skills_list"])
        # Also recognize skills appearing in the profile summary or education.
        candidate_skills.update(extract_skills(
            f"{candidate['profile_summary']} {candidate['education']}"
        ))
        required_skills.update(extract_skills(job["description"]))

        matched = sorted(required_skills & candidate_skills)
        missing = sorted(required_skills - candidate_skills)
        skill_ratio = len(matched) / len(required_skills) if required_skills else 0.0

        required_exp = max(float(job["experience_required"]), 0.0)
        candidate_exp = max(float(candidate["experience_years"]), 0.0)
        exp_ratio = min(candidate_exp / required_exp, 1.0) if required_exp else 1.0

        total = (similarity * 0.60 + skill_ratio * 0.30 + exp_ratio * 0.10) * 100
        return {
            "match_score": round(total, 2),
            "text_similarity": round(similarity * 100, 2),
            "skill_match": round(skill_ratio * 100, 2),
            "experience_match": round(exp_ratio * 100, 2),
            "matched_skills": matched,
            "missing_skills": missing,
        }

    def match_candidate_to_jobs(self, candidate_id, top_n=None):
        """Return jobs ranked by suitability for the selected candidate."""
        rows = self.candidates_df[self.candidates_df["candidate_id"] == candidate_id]
        if rows.empty:
            return None
        candidate_idx = int(rows.index[0])
        matches = []
        for job_idx, job in self.jobs_df.iterrows():
            score = self.calculate_match_score(job_idx, candidate_idx)
            matches.append({
                "job_id": job["job_id"], "title": job["title"],
                "company": job["company"], "location": job["location"],
                "salary_range": job["salary_range"], **score
            })
        matches.sort(key=lambda item: item["match_score"], reverse=True)
        return matches[:top_n] if top_n else matches

    def rank_candidates_for_job(self, job_id, top_n=None):
        """Return candidates ranked by suitability for the selected job."""
        rows = self.jobs_df[self.jobs_df["job_id"] == job_id]
        if rows.empty:
            return None
        job_idx = int(rows.index[0])
        ranked = []
        for candidate_idx, candidate in self.candidates_df.iterrows():
            score = self.calculate_match_score(job_idx, candidate_idx)
            ranked.append({
                "candidate_id": candidate["candidate_id"],
                "name": candidate["name"],
                "experience": candidate["experience_years"],
                "education": candidate["education"],
                "preferred_location": candidate["preferred_location"],
                **score
            })
        ranked.sort(key=lambda item: item["match_score"], reverse=True)
        return ranked[:top_n] if top_n else ranked

    def get_job_details(self, job_id):
        rows = self.jobs_df[self.jobs_df["job_id"] == job_id]
        return None if rows.empty else rows.iloc[0].to_dict()

    def get_candidate_details(self, candidate_id):
        rows = self.candidates_df[self.candidates_df["candidate_id"] == candidate_id]
        return None if rows.empty else rows.iloc[0].to_dict()

    def save_all_matches(self, output_path=None):
        """Save every job-candidate match to a CSV report."""
        records = []
        for job_idx, job in self.jobs_df.iterrows():
            for candidate_idx, candidate in self.candidates_df.iterrows():
                score = self.calculate_match_score(job_idx, candidate_idx)
                records.append({
                    "job_id": job["job_id"], "job_title": job["title"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_name": candidate["name"],
                    **{key: value for key, value in score.items()
                       if key not in ("matched_skills", "missing_skills")},
                    "matched_skills": "|".join(score["matched_skills"]),
                    "missing_skills": "|".join(score["missing_skills"]),
                })
        report = pd.DataFrame(records).sort_values(
            ["job_id", "match_score"], ascending=[True, False]
        )
        destination = Path(output_path) if output_path else BASE_DIR / "reports" / "all_matches.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(destination, index=False)
        return destination


def print_job_matches(engine, candidate_id):
    matches = engine.match_candidate_to_jobs(candidate_id)
    if matches is None:
        print(f"Candidate '{candidate_id}' was not found.")
        return
    candidate = engine.get_candidate_details(candidate_id)
    print(f"\nJOB RECOMMENDATIONS FOR {candidate['name']} ({candidate_id})")
    print("=" * 72)
    for rank, item in enumerate(matches, 1):
        print(f"\n{rank}. {item['title']} — {item['company']} [{item['job_id']}]")
        print(f"   Match score: {item['match_score']}% | Skills: {item['skill_match']}% | "
              f"Experience: {item['experience_match']}%")
        print(f"   Location: {item['location']} | Salary: {item['salary_range']}")
        print("   Matched skills:", ", ".join(item["matched_skills"]) or "None")
        print("   Skill gaps:    ", ", ".join(item["missing_skills"]) or "None")


def print_candidate_ranking(engine, job_id):
    ranked = engine.rank_candidates_for_job(job_id)
    if ranked is None:
        print(f"Job '{job_id}' was not found.")
        return
    job = engine.get_job_details(job_id)
    print(f"\nCANDIDATE RANKING FOR {job['title']} ({job_id})")
    print("=" * 72)
    for rank, item in enumerate(ranked, 1):
        print(f"\n{rank}. {item['name']} ({item['candidate_id']})")
        print(f"   Match score: {item['match_score']}% | Skills: {item['skill_match']}% | "
              f"Experience: {item['experience_match']}%")
        print(f"   Experience: {item['experience']} years | Education: {item['education']}")
        print("   Matched skills:", ", ".join(item["matched_skills"]) or "None")
        print("   Skill gaps:    ", ", ".join(item["missing_skills"]) or "None")


def show_data(engine):
    print("\nAVAILABLE JOBS")
    print(engine.jobs_df[["job_id", "title", "company", "experience_required", "location"]]
          .to_string(index=False))
    print("\nAVAILABLE CANDIDATES")
    print(engine.candidates_df[
        ["candidate_id", "name", "experience_years", "education", "preferred_location"]
    ].to_string(index=False))


def main():
    try:
        engine = JobMatchingEngine()
    except Exception as exc:
        print("Could not load the project data:", exc)
        sys.exit(1)

    while True:
        print("\n" + "=" * 60)
        print("          AI JOB MATCHING ENGINE | AI-RH-005")
        print("=" * 60)
        print("1. Show available jobs and candidates")
        print("2. Recommend jobs for a candidate")
        print("3. Rank candidates for a job")
        print("4. View candidate skill gap for a job")
        print("5. Export all match scores to CSV")
        print("0. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            show_data(engine)
        elif choice == "2":
            candidate_id = input("Enter candidate ID (e.g., C001): ").strip().upper()
            print_job_matches(engine, candidate_id)
        elif choice == "3":
            job_id = input("Enter job ID (e.g., J001): ").strip().upper()
            print_candidate_ranking(engine, job_id)
        elif choice == "4":
            candidate_id = input("Enter candidate ID (e.g., C001): ").strip().upper()
            job_id = input("Enter job ID (e.g., J001): ").strip().upper()
            matches = engine.match_candidate_to_jobs(candidate_id)
            if matches is None:
                print(f"Candidate '{candidate_id}' was not found.")
                continue
            result = next((item for item in matches if item["job_id"] == job_id), None)
            if result is None:
                print(f"Job '{job_id}' was not found.")
            else:
                print(f"\n{candidate_id} -> {job_id}: {result['match_score']}% match")
                print("Matched skills:", ", ".join(result["matched_skills"]) or "None")
                print("Skills to develop:", ", ".join(result["missing_skills"]) or "None")
        elif choice == "5":
            path = engine.save_all_matches()
            print(f"All match results saved to: {path}")
        elif choice == "0":
            print("Thank you for using AI Job Matching Engine.")
            break
        else:
            print("Invalid choice. Please enter 0, 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
