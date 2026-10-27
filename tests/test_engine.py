import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import JobMatchingEngine, extract_skills, preprocess_text


class TestJobMatchingEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = JobMatchingEngine()

    def test_preprocess_text(self):
        self.assertIn("python", preprocess_text("Python, DATA!"))

    def test_extract_skills(self):
        skills = extract_skills("I use Python, TensorFlow and machine learning.")
        self.assertIn("python", skills)
        self.assertIn("tensorflow", skills)

    def test_candidate_recommendations(self):
        results = self.engine.match_candidate_to_jobs("C001")
        self.assertEqual(len(results), len(self.engine.jobs_df))
        self.assertGreaterEqual(results[0]["match_score"], results[-1]["match_score"])

    def test_candidate_ranking(self):
        results = self.engine.rank_candidates_for_job("J001")
        self.assertEqual(len(results), len(self.engine.candidates_df))
        self.assertGreaterEqual(results[0]["match_score"], results[-1]["match_score"])

    def test_unknown_ids(self):
        self.assertIsNone(self.engine.match_candidate_to_jobs("C999"))
        self.assertIsNone(self.engine.rank_candidates_for_job("J999"))


if __name__ == "__main__":
    unittest.main()
