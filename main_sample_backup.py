# AI Job Matching Engine - NLP & Machine Learning
# Free Online AI & Data Science Internship Task AI-RH-005

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class JobMatchingEngine:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.jobs_df = self.load_jobs()
        self.candidates_df = self.load_candidates()
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.job_vectors = None
        self.candidate_vectors = None

        # Skill categories for matching
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'sql', 'r'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'django', 'flask'],
            'data_science': ['machine learning', 'deep learning', 'nlp',
                             'computer vision', 'tensorflow', 'pytorch',
                             'pandas', 'numpy', 'scikit-learn'],
            'database': ['mysql', 'postgresql', 'mongodb', 'oracle', 'redis'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'soft_skills': ['communication', 'teamwork', 'leadership',
                            'problem solving', 'time management', 'adaptability']
        }

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def extract_skills(self, text):
        """Extract skills from text"""
        text_lower = text.lower()
        found_skills = []

        for category, skills in self.skill_categories.items():
            for skill in skills:
                if skill in text_lower:
                    found_skills.append(skill)

        return found_skills

    def load_jobs(self):
        """Load job database"""
        data = {
            'job_id': ['J001', 'J002', 'J003', 'J004', 'J005',
                       'J006', 'J007', 'J008'],
            'title': [
                'AI/ML Engineer',
                'Data Scientist',
                'Full Stack Developer',
                'DevOps Engineer',
                'Cyber Security Analyst',
                'Data Analyst',
                'Backend Developer',
                'Product Manager'
            ],
            'company': [
                'TechCorp', 'DataWorks', 'WebSolutions',
                'CloudOps', 'SecureNet', 'InsightAI',
                'CodeBase', 'ProductLabs'
            ],
            'description': [
                "Build and deploy machine learning models using Python, TensorFlow, PyTorch. Experience with NLP, computer vision, and deep learning required.",
                "Analyze complex datasets to derive insights using Python, R, SQL, statistics, and data visualization. Machine learning experience preferred.",
                "Develop complete web applications using JavaScript, React, Node.js, Python, SQL, HTML, CSS. Full stack development experience required.",
                "Manage cloud infrastructure and CI/CD pipelines using AWS, Docker, Kubernetes, Jenkins, Python, Linux. DevOps experience required.",
                "Protect systems and networks using Python, network security, cryptography, penetration testing, Linux. Cyber security experience required.",
                "Interpret data and create reports using SQL, Excel, Python, data visualization, statistics. Strong analytical skills required.",
                "Build server-side applications using Python, Django, Flask, SQL, REST APIs, Docker. Backend development experience required.",
                "Lead product development using product strategy, market research, agile, communication, data analysis. Product management experience required."
            ],
            'required_skills': [
                ['python', 'tensorflow', 'pytorch', 'machine learning', 'deep learning'],
                ['python', 'r', 'sql', 'statistics', 'machine learning'],
                ['javascript', 'react', 'node.js', 'python', 'sql'],
                ['aws', 'docker', 'kubernetes', 'python', 'linux'],
                ['python', 'network security', 'cryptography', 'linux'],
                ['sql', 'excel', 'python', 'data visualization'],
                ['python', 'django', 'flask', 'sql', 'rest apis'],
                ['product strategy', 'market research', 'agile', 'communication']
            ],
            'experience_required': [3, 2, 2, 3, 2, 1, 2, 4],
            'location': [
                'Remote', 'Remote', 'Hybrid', 'Remote',
                'On-site', 'Hybrid', 'Remote', 'Hybrid'
            ],
            'salary_range': [
                '₹8-25 LPA', '₹10-30 LPA', '₹8-22 LPA', '₹8-25 LPA',
                '₹6-20 LPA', '₹5-15 LPA', '₹6-18 LPA', '₹12-35 LPA'
            ]
        }
        df = pd.DataFrame(data)
        df['combined_text'] = df['title'] + ' ' + df['description'] + ' ' + df['required_skills'].apply(
            lambda x: ' '.join(x))
        df['clean_text'] = df['combined_text'].apply(self.preprocess_text)
        return df

    def load_candidates(self):
        """Load candidate profiles"""
        data = {
            'candidate_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown',
                     'Diana Prince', 'Eve Wilson'],
            'skills': [
                ['python', 'tensorflow', 'machine learning', 'deep learning', 'nlp'],
                ['python', 'sql', 'statistics', 'data visualization', 'r'],
                ['javascript', 'react', 'node.js', 'python', 'sql'],
                ['aws', 'docker', 'kubernetes', 'python', 'linux'],
                ['python', 'network security', 'cryptography', 'linux']
            ],
            'experience_years': [3, 2, 2, 4, 3],
            'education': [
                'M.Tech in AI',
                'B.Tech in Data Science',
                'B.Tech in Computer Science',
                'B.Tech in IT',
                'M.Tech in Cyber Security'
            ],
            'preferred_location': ['Remote', 'Remote', 'Hybrid', 'Remote', 'On-site'],
            'expected_salary': ['₹15 LPA', '₹12 LPA', '₹10 LPA', '₹18 LPA', '₹14 LPA']
        }
        df = pd.DataFrame(data)
        df['combined_text'] = df['skills'].apply(lambda x: ' '.join(x)) + ' ' + df['education']
        df['clean_text'] = df['combined_text'].apply(self.preprocess_text)
        return df

    def create_vectors(self):
        """Create TF-IDF vectors for jobs and candidates"""
        all_text = list(self.jobs_df['clean_text']) + list(self.candidates_df['clean_text'])
        self.vectorizer.fit(all_text)

        self.job_vectors = self.vectorizer.transform(self.jobs_df['clean_text'])
        self.candidate_vectors = self.vectorizer.transform(self.candidates_df['clean_text'])

    def calculate_match_score(self, job_idx, candidate_idx):
        """Calculate match score between a job and a candidate"""
        if self.job_vectors is None:
            self.create_vectors()

        # Text similarity score (60% weight)
        text_similarity = cosine_similarity(
            self.job_vectors[job_idx:job_idx + 1],
            self.candidate_vectors[candidate_idx:candidate_idx + 1]
        )[0][0]

        # Skill match score (30% weight)
        job_skills = set(self.jobs_df.iloc[job_idx]['required_skills'])
        candidate_skills = set(self.candidates_df.iloc[candidate_idx]['skills'])

        if len(job_skills) > 0:
            skill_match = len(job_skills & candidate_skills) / len(job_skills)
        else:
            skill_match = 0

        # Experience match score (10% weight)
        required_exp = self.jobs_df.iloc[job_idx]['experience_required']
        candidate_exp = self.candidates_df.iloc[candidate_idx]['experience_years']

        if required_exp > 0:
            exp_match = min(candidate_exp / required_exp, 1.0)
        else:
            exp_match = 1.0

        # Calculate weighted score
        match_score = (
                              text_similarity * 0.6 +
                              skill_match * 0.3 +
                              exp_match * 0.1
                      ) * 100

        return {
            'match_score': round(match_score, 2),
            'text_similarity': round(text_similarity * 100, 2),
            'skill_match': round(skill_match * 100, 2),
            'experience_match': round(exp_match * 100, 2),
            'matched_skills': list(job_skills & candidate_skills),
            'missing_skills': list(job_skills - candidate_skills)
        }

    def match_candidate_to_jobs(self, candidate_id, top_n=3):
        """Find best job matches for a candidate"""
        if candidate_id not in self.candidates_df['candidate_id'].values:
            return "Candidate not found"

        candidate_idx = self.candidates_df[self.candidates_df['candidate_id'] == candidate_id].index[0]

        matches = []
        for job_idx in range(len(self.jobs_df)):
            match_result = self.calculate_match_score(job_idx, candidate_idx)
            job = self.jobs_df.iloc[job_idx]

            matches.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'company': job['company'],
                'location': job['location'],
                'salary_range': job['salary_range'],
                'match_score': match_result['match_score'],
                'skill_match': match_result['skill_match'],
                'matched_skills': match_result['matched_skills'],
                'missing_skills': match_result['missing_skills']
            })

        # Sort by match score and return top N
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches[:top_n]

    def rank_candidates_for_job(self, job_id, top_n=3):
        """Rank candidates for a specific job"""
        if job_id not in self.jobs_df['job_id'].values:
            return "Job not found"

        job_idx = self.jobs_df[self.jobs_df['job_id'] == job_id].index[0]

        candidates = []
        for candidate_idx in range(len(self.candidates_df)):
            match_result = self.calculate_match_score(job_idx, candidate_idx)
            candidate = self.candidates_df.iloc[candidate_idx]

            candidates.append({
                'candidate_id': candidate['candidate_id'],
                'name': candidate['name'],
                'experience': candidate['experience_years'],
                'education': candidate['education'],
                'match_score': match_result['match_score'],
                'skill_match': match_result['skill_match'],
                'matched_skills': match_result['matched_skills']
            })

        candidates.sort(key=lambda x: x['match_score'], reverse=True)
        return candidates[:top_n]

    def get_job_details(self, job_id):
        """Get detailed information about a job"""
        job = self.jobs_df[self.jobs_df['job_id'] == job_id]
        if len(job) == 0:
            return "Job not found"
        return job.iloc[0].to_dict()

    def get_candidate_details(self, candidate_id):
        """Get detailed information about a candidate"""
        candidate = self.candidates_df[self.candidates_df['candidate_id'] == candidate_id]
        if len(candidate) == 0:
            return "Candidate not found"
        return candidate.iloc[0].to_dict()


# Usage Example
engine = JobMatchingEngine()

# Match candidate to jobs
print("🎯 Job Matches for Candidate C001 (Alice Johnson):")
print("=" * 60)

matches = engine.match_candidate_to_jobs('C001', top_n=3)

for i, match in enumerate(matches, 1):
    print(f"\n{i}. {match['title']} at {match['company']}")
    print(f"   📍 Location: {match['location']}")
    print(f"   💰 Salary: {match['salary_range']}")
    print(f"   📊 Match Score: {match['match_score']}%")
    print(f"   🔧 Skill Match: {match['skill_match']}%")
    print(f"   ✅ Matched Skills: {', '.join(match['matched_skills'])}")
    if match['missing_skills']:
        print(f"   ⚠️ Missing Skills: {', '.join(match['missing_skills'])}")

# Rank candidates for a job
print("\n\n📋 Top Candidates for Job J001 (AI/ML Engineer):")
print("=" * 60)

candidates = engine.rank_candidates_for_job('J001', top_n=3)

for i, candidate in enumerate(candidates, 1):
    print(f"\n{i}. {candidate['name']} ({candidate['candidate_id']})")
    print(f"   🎓 Education: {candidate['education']}")
    print(f"   💼 Experience: {candidate['experience']} years")
    print(f"   📊 Match Score: {candidate['match_score']}%")
    print(f"   ✅ Matched Skills: {', '.join(candidate['matched_skills'])}")