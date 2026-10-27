
import os
import re
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Job Matching Engine",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. BLUE, GREEN AND WHITE THEME
# --------------------------------------------------

st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #0b1f3a 0%,
        #075e54 55%,
        #123b68 100%
    );
    color: #ffffff;
}

/* Headings */
h1, h2, h3, h4 {
    color: #ffffff !important;
}

/* General text */
.stApp p,
.stApp label,
.stMarkdown,
[data-testid="stCaptionContainer"] {
    color: #f0fdf4;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0a2038;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.10);
    border: 1px solid #34d399;
    padding: 18px;
    border-radius: 14px;
}

/* Metric labels and values */
[data-testid="stMetricLabel"] {
    color: #d1fae5 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background-color: #10b981;
    color: #ffffff;
    border: 1px solid #6ee7b7;
    border-radius: 10px;
}

/* Button hover */
.stButton button:hover,
.stDownloadButton button:hover {
    background-color: #059669;
    color: #ffffff;
    border-color: #ffffff;
}

/* Progress bars */
.stProgress > div > div > div {
    background-color: #34d399;
}

/* Dropdowns */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #ffffff;
    border-color: #34d399;
    color: #0b1f3a;
}

/* Text and number input fields */
.stTextInput input,
.stNumberInput input {
    background-color: #ffffff;
    color: #0b1f3a;
}

/* Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #34d399;
    border-radius: 14px;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border: 1px solid #34d399;
    border-radius: 10px;
}

/* Links */
a {
    color: #6ee7b7 !important;
}

/* Divider */
hr {
    border-color: #34d399;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CANDIDATES_FILE = os.path.join(DATA_DIR, "candidates.csv")
JOBS_FILE = os.path.join(DATA_DIR, "jobs.csv")


# --------------------------------------------------
# 4. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    candidates = pd.read_csv(CANDIDATES_FILE)
    jobs = pd.read_csv(JOBS_FILE)

    return candidates, jobs


st.title("💼 AI Job Matching Engine")
st.markdown(
    "### Find suitable jobs using skills, experience, and profile similarity."
)

try:
    candidates, jobs = load_data()

except FileNotFoundError:
    st.error(
        "CSV files were not found. Please check that "
        "'candidates.csv' and 'jobs.csv' are inside the data folder."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the CSV files: {error}")
    st.stop()


# --------------------------------------------------
# 5. CHECK REQUIRED COLUMNS
# --------------------------------------------------

candidate_columns = [
    "candidate_id",
    "name",
    "skills",
    "experience_years",
    "education",
    "preferred_location",
    "expected_salary",
    "profile_summary"
]

job_columns = [
    "job_id",
    "title",
    "company",
    "description",
    "required_skills",
    "experience_required",
    "location",
    "salary_range"
]

missing_candidate_columns = [
    column for column in candidate_columns
    if column not in candidates.columns
]

missing_job_columns = [
    column for column in job_columns
    if column not in jobs.columns
]

if missing_candidate_columns:
    st.error(
        "Missing columns in candidates.csv: "
        + ", ".join(missing_candidate_columns)
    )
    st.stop()

if missing_job_columns:
    st.error(
        "Missing columns in jobs.csv: "
        + ", ".join(missing_job_columns)
    )
    st.stop()


# --------------------------------------------------
# 6. CLEAN TEXT AND NUMBERS
# --------------------------------------------------

def clean_text(value):
    """Convert a value into clean lowercase text."""

    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_skills(value):
    """Convert a comma-separated skill list into a set."""

    if pd.isna(value):
        return set()

    skills = str(value).lower().split(",")

    return {
        clean_text(skill)
        for skill in skills
        if clean_text(skill)
    }


def to_number(value):
    """Convert a value into a number safely."""

    try:
        if pd.isna(value):
            return 0.0

        match = re.search(r"-?\d+(\.\d+)?", str(value))

        if match:
            return float(match.group())

        return 0.0

    except (ValueError, TypeError):
        return 0.0


def get_location(value):
    """Return a normalized location."""

    return clean_text(value)


# --------------------------------------------------
# 7. PREPARE TEXT FOR MATCHING
# --------------------------------------------------

candidates = candidates.copy()
jobs = jobs.copy()

candidates["skills"] = candidates["skills"].fillna("")
candidates["profile_summary"] = (
    candidates["profile_summary"].fillna("")
)
candidates["preferred_location"] = (
    candidates["preferred_location"].fillna("")
)
candidates["experience_years"] = (
    candidates["experience_years"].apply(to_number)
)
candidates["expected_salary"] = (
    candidates["expected_salary"].apply(to_number)
)

jobs["required_skills"] = jobs["required_skills"].fillna("")
jobs["description"] = jobs["description"].fillna("")
jobs["location"] = jobs["location"].fillna("")
jobs["experience_required"] = (
    jobs["experience_required"].apply(to_number)
)

candidates["combined_text"] = (
    candidates["skills"].apply(clean_text)
    + " "
    + candidates["profile_summary"].apply(clean_text)
    + " "
    + candidates["education"].apply(clean_text)
)

jobs["combined_text"] = (
    jobs["required_skills"].apply(clean_text)
    + " "
    + jobs["description"].apply(clean_text)
    + " "
    + jobs["title"].apply(clean_text)
)


# --------------------------------------------------
# 8. CALCULATE MATCH SCORES
# --------------------------------------------------

def calculate_matches(candidates_df, jobs_df):
    """Calculate a match score for every candidate-job pair."""

    if candidates_df.empty or jobs_df.empty:
        return pd.DataFrame()

    all_text = (
        candidates_df["combined_text"].tolist()
        + jobs_df["combined_text"].tolist()
    )

    # TF-IDF converts text into numerical vectors.
    vectorizer = TfidfVectorizer()

    try:
        text_vectors = vectorizer.fit_transform(all_text)

    except ValueError:
        # Handles cases where all text is empty.
        text_vectors = None

    candidate_count = len(candidates_df)

    if text_vectors is not None:
        candidate_vectors = text_vectors[:candidate_count]
        job_vectors = text_vectors[candidate_count:]

        similarities = cosine_similarity(
            candidate_vectors,
            job_vectors
        )
    else:
        similarities = [
            [0.0] * len(jobs_df)
            for _ in range(len(candidates_df))
        ]

    results = []

    for candidate_index, (_, candidate) in enumerate(
        candidates_df.iterrows()
    ):
        candidate_skills = get_skills(candidate["skills"])

        for job_index, (_, job) in enumerate(jobs_df.iterrows()):
            job_skills = get_skills(job["required_skills"])

            # Skill matching score
            if job_skills:
                matched_skills = candidate_skills & job_skills

                skill_score = (
                    len(matched_skills) / len(job_skills)
                ) * 100
            else:
                matched_skills = set()
                skill_score = 0.0

            # Text similarity score
            if text_vectors is not None:
                text_score = (
                    float(similarities[candidate_index][job_index])
                    * 100
                )
            else:
                text_score = 0.0

            # Experience score
            candidate_experience = to_number(
                candidate["experience_years"]
            )
            required_experience = to_number(
                job["experience_required"]
            )

            if required_experience <= 0:
                experience_score = 100.0
            else:
                experience_score = min(
                    candidate_experience / required_experience,
                    1.0
                ) * 100

            # Weighted overall score
            overall_score = (
                text_score * 0.60
                + skill_score * 0.30
                + experience_score * 0.10
            )

            missing_skills = job_skills - candidate_skills

            results.append({
                "candidate_id": candidate["candidate_id"],
                "candidate_name": candidate["name"],
                "job_id": job["job_id"],
                "job_title": job["title"],
                "company": job["company"],
                "job_location": job["location"],
                "salary_range": job["salary_range"],
                "candidate_experience": candidate_experience,
                "required_experience": required_experience,
                "text_similarity": round(text_score, 2),
                "skill_score": round(skill_score, 2),
                "experience_score": round(experience_score, 2),
                "match_score": round(overall_score, 2),
                "matched_skills": ", ".join(
                    sorted(matched_skills)
                ),
                "missing_skills": ", ".join(
                    sorted(missing_skills)
                )
            })

    return pd.DataFrame(results)


with st.spinner("Calculating job matches..."):
    matches = calculate_matches(candidates, jobs)


# --------------------------------------------------
# 9. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Dashboard",
        "Find Jobs",
        "Rank Candidates",
        "Skill Gap Analysis",
        "All Matches"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This system uses TF-IDF text similarity, "
    "skill overlap, and experience matching."
)


# --------------------------------------------------
# 10. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Candidates",
            len(candidates)
        )

    with col2:
        st.metric(
            "Available Jobs",
            len(jobs)
        )

    with col3:
        st.metric(
            "Possible Matches",
            len(matches)
        )

    with col4:
        strong_matches = (
            int((matches["match_score"] >= 70).sum())
            if not matches.empty
            else 0
        )

        st.metric(
            "Strong Matches",
            strong_matches
        )

    st.markdown("---")

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("💼 Available Jobs")

        if jobs.empty:
            st.info("No jobs are available.")
        else:
            st.dataframe(
                jobs[
                    ["job_id", "title", "company", "location"]
                ],
                use_container_width=True,
                hide_index=True
            )

    with right_column:
        st.subheader("👤 Registered Candidates")

        if candidates.empty:
            st.info("No candidates are available.")
        else:
            st.dataframe(
                candidates[
                    [
                        "candidate_id",
                        "name",
                        "skills",
                        "experience_years"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---")
    st.subheader("📈 Match Score Distribution")

    if not matches.empty:
        chart_data = matches[["match_score"]].copy()
        chart_data["Score Range"] = pd.cut(
            chart_data["match_score"],
            bins=[-1, 20, 40, 60, 80, 100],
            labels=[
                "0–20",
                "21–40",
                "41–60",
                "61–80",
                "81–100"
            ]
        )

        distribution = (
            chart_data["Score Range"]
            .value_counts(sort=False)
        )

        st.bar_chart(distribution)

    else:
        st.info("There are no matches to display.")


# --------------------------------------------------
# 11. FIND JOBS FOR A CANDIDATE
# --------------------------------------------------

elif page == "Find Jobs":

    st.header("🔎 Find Suitable Jobs")

    if candidates.empty:
        st.warning("No candidates are available.")
        st.stop()

    candidate_names = candidates["name"].astype(str).tolist()

    selected_candidate_name = st.selectbox(
        "Select a candidate",
        candidate_names
    )

    selected_candidate = candidates[
        candidates["name"].astype(str) == selected_candidate_name
    ].iloc[0]

    st.markdown("### Candidate Profile")

    st.write(
        f"**Name:** {selected_candidate['name']}"
    )
    st.write(
        f"**Skills:** {selected_candidate['skills']}"
    )
    st.write(
        f"**Experience:** "
        f"{selected_candidate['experience_years']} years"
    )
    st.write(
        f"**Education:** {selected_candidate['education']}"
    )
    st.write(
        f"**Preferred Location:** "
        f"{selected_candidate['preferred_location']}"
    )

    candidate_matches = matches[
        matches["candidate_id"] == selected_candidate["candidate_id"]
    ].sort_values(
        by="match_score",
        ascending=False
    )

    st.markdown("---")
    st.subheader("🎯 Recommended Jobs")

    minimum_score = st.slider(
        "Minimum match score",
        min_value=0,
        max_value=100,
        value=0,
        step=5
    )

    filtered_matches = candidate_matches[
        candidate_matches["match_score"] >= minimum_score
    ]

    if filtered_matches.empty:
        st.info(
            "No jobs meet this score. Try lowering the minimum score."
        )

    else:
        for _, result in filtered_matches.iterrows():

            with st.container(border=True):

                st.subheader(
                    f"{result['job_title']} at {result['company']}"
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.write(
                        f"📍 **Location:** {result['job_location']}"
                    )
                    st.write(
                        f"💰 **Salary:** {result['salary_range']}"
                    )

                with col2:
                    st.write(
                        f"⭐ **Match Score:** "
                        f"{result['match_score']}%"
                    )
                    st.progress(
                        int(result["match_score"]) / 100
                    )

                st.write(
                    f"**Matched Skills:** "
                    f"{result['matched_skills'] or 'None identified'}"
                )

                st.write(
                    f"**Skills to Improve:** "
                    f"{result['missing_skills'] or 'None identified'}"
                )

                with st.expander("View score details"):
                    st.write(
                        f"Text similarity: "
                        f"{result['text_similarity']}%"
                    )
                    st.write(
                        f"Skill score: {result['skill_score']}%"
                    )
                    st.write(
                        f"Experience score: "
                        f"{result['experience_score']}%"
                    )


# --------------------------------------------------
# 12. RANK CANDIDATES FOR A JOB
# --------------------------------------------------

elif page == "Rank Candidates":

    st.header("🏆 Rank Candidates")

    if jobs.empty:
        st.warning("No jobs are available.")
        st.stop()

    selected_job_title = st.selectbox(
        "Select a job",
        jobs["title"].astype(str).tolist()
    )

    selected_job = jobs[
        jobs["title"].astype(str) == selected_job_title
    ].iloc[0]

    st.write(f"**Company:** {selected_job['company']}")
    st.write(f"**Required Skills:** {selected_job['required_skills']}")
    st.write(
        f"**Required Experience:** "
        f"{selected_job['experience_required']} years"
    )

    job_matches = matches[
        matches["job_id"] == selected_job["job_id"]
    ].sort_values(
        by="match_score",
        ascending=False
    )

    st.subheader("Candidate Ranking")

    if job_matches.empty:
        st.info("No candidates can be ranked for this job.")

    else:
        display_ranking = job_matches[
            [
                "candidate_name",
                "candidate_experience",
                "match_score",
                "matched_skills",
                "missing_skills"
            ]
        ].copy()

        display_ranking.insert(
            0,
            "Rank",
            range(1, len(display_ranking) + 1)
        )

        display_ranking = display_ranking.rename(
            columns={
                "candidate_name": "Candidate",
                "candidate_experience": "Experience (Years)",
                "match_score": "Match Score (%)",
                "matched_skills": "Matched Skills",
                "missing_skills": "Missing Skills"
            }
        )

        st.dataframe(
            display_ranking,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Top Candidate")

        top_candidate = job_matches.iloc[0]

        st.success(
            f"Top match: {top_candidate['candidate_name']} "
            f"with a score of {top_candidate['match_score']}%."
        )

        st.caption(
            "This is a ranking based on the configured matching "
            "formula, not a guarantee of job suitability."
        )


# --------------------------------------------------
# 13. SKILL GAP ANALYSIS
# --------------------------------------------------

elif page == "Skill Gap Analysis":

    st.header("🧠 Skill Gap Analysis")

    if candidates.empty:
        st.warning("No candidates are available.")
        st.stop()

    selected_candidate_name = st.selectbox(
        "Choose a candidate",
        candidates["name"].astype(str).tolist(),
        key="skill_gap_candidate"
    )

    selected_candidate = candidates[
        candidates["name"].astype(str) == selected_candidate_name
    ].iloc[0]

    candidate_skills = get_skills(
        selected_candidate["skills"]
    )

    st.subheader("Current Skills")

    if candidate_skills:
        st.write(", ".join(sorted(candidate_skills)))
    else:
        st.info("No candidate skills have been entered.")

    st.markdown("---")

    selected_job_title = st.selectbox(
        "Choose a job to compare",
        jobs["title"].astype(str).tolist(),
        key="skill_gap_job"
    )

    selected_job = jobs[
        jobs["title"].astype(str) == selected_job_title
    ].iloc[0]

    required_skills = get_skills(
        selected_job["required_skills"]
    )

    matched_skills = candidate_skills & required_skills
    missing_skills = required_skills - candidate_skills

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Required Skills",
            len(required_skills)
        )

    with col2:
        st.metric(
            "Matched Skills",
            len(matched_skills)
        )

    with col3:
        st.metric(
            "Missing Skills",
            len(missing_skills)
        )

    st.subheader("✅ Skills You Have")

    if matched_skills:
        for skill in sorted(matched_skills):
            st.write(f"🟢 {skill}")
    else:
        st.info("No exact skill matches were found.")

    st.subheader("📚 Skills to Learn")

    if missing_skills:
        for skill in sorted(missing_skills):
            st.write(f"🔵 {skill}")
    else:
        st.success(
            "All listed job skills match the candidate's listed skills."
        )

    if required_skills:
        coverage = (
            len(matched_skills) / len(required_skills)
        ) * 100

        st.subheader("Skill Coverage")
        st.progress(coverage / 100)
        st.write(f"Required skill coverage: {coverage:.1f}%")

    st.caption(
        "Skill gaps are based on exact, comma-separated skill names. "
        "Similar terms such as 'ML' and 'Machine Learning' may not "
        "be treated as identical."
    )


# --------------------------------------------------
# 14. ALL MATCHES
# --------------------------------------------------

elif page == "All Matches":

    st.header("📋 All Candidate–Job Matches")

    if matches.empty:
        st.info("No matches are available.")
        st.stop()

    minimum_score = st.slider(
        "Filter by minimum match score",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        key="all_matches_minimum"
    )

    display_matches = matches[
        matches["match_score"] >= minimum_score
    ].sort_values(
        by="match_score",
        ascending=False
    )

    st.write(
        f"Showing {len(display_matches)} matches."
    )

    columns_to_display = [
        "candidate_name",
        "job_title",
        "company",
        "job_location",
        "match_score",
        "matched_skills",
        "missing_skills"
    ]

    display_table = display_matches[
        columns_to_display
    ].rename(
        columns={
            "candidate_name": "Candidate",
            "job_title": "Job Title",
            "company": "Company",
            "job_location": "Location",
            "match_score": "Match Score (%)",
            "matched_skills": "Matched Skills",
            "missing_skills": "Missing Skills"
        }
    )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )

    # Download the filtered results.
    csv_data = display_table.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Matches as CSV",
        data=csv_data,
        file_name="job_matching_results.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 15. FOOTER
# --------------------------------------------------

st.markdown("---")git status

st.caption(
    "AI Job Matching Engine | Python • Streamlit • Pandas • "
    "Scikit-learn"
)
