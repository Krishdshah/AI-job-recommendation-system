import json
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load jobs and sample resume
@st.cache_data
def load_jobs():
    with open("jobs_task2.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_sample_resume():
    with open("sample_resume.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

jobs = load_jobs()
model = load_model()

# Looser semantic similarity
def semantic_score(list1, list2):
    if not list1 or not list2:
        return 0
    emb1 = model.encode(list1, convert_to_tensor=True)
    emb2 = model.encode(list2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).mean().cpu().numpy()  # mean instead of max
    return float(score)

# Looser exact match (case-insensitive + partials)
def exact_match_score(list1, list2):
    if not list1 or not list2:
        return 0
    list1_lower = [x.lower() for x in list1]
    list2_lower = [x.lower() for x in list2]
    matches = 0
    for a in list1_lower:
        for b in list2_lower:
            if a in b or b in a:  # partial match
                matches += 1
    return matches / len(list1_lower)

# Salary match
def salary_score(candidate_min, job_range):
    if not job_range or len(job_range) != 2:
        return 0
    if job_range[1] < candidate_min:
        return 0
    if job_range[0] >= candidate_min:
        return 1
    return 0.7  # partial credit for overlap

# Weighted scoring
def compute_match(job, prefs):
    weights = {
        "skills": 0.30,
        "title": 0.20,
        "location": 0.15,
        "industry": 0.10,
        "company_size": 0.10,
        "values": 0.10,
        "salary": 0.05
    }

    score_skills = semantic_score(prefs.get("skills", []), job.get("required_skills", []))
    score_title = semantic_score(prefs.get("titles", []), [job.get("title", "")]) + \
                  exact_match_score(prefs.get("role_types", []), [job.get("employment_type", "")])
    score_title = min(score_title / 2, 1)

    score_location = exact_match_score(prefs.get("locations", []), [job.get("location", "")])
    score_industry = exact_match_score(prefs.get("industries", []), [job.get("industry", "")])
    score_company = exact_match_score(prefs.get("company_size", []), [job.get("company_size", "")])
    score_values = exact_match_score(prefs.get("values", []), job.get("values_promoted", []))
    score_salary = salary_score(prefs.get("min_salary", 0), job.get("salary_range", []))

    final_score = (
        score_skills * weights["skills"] +
        score_title * weights["title"] +
        score_location * weights["location"] +
        score_industry * weights["industry"] +
        score_company * weights["company_size"] +
        score_values * weights["values"] +
        score_salary * weights["salary"]
    )

    return round(final_score * 100, 2)

# UI
st.title("🎯 Weighted AI Job Recommendation System")

mode = st.radio("Choose input method", ["Use sample resume file", "Fill out form"])

if mode == "Use sample resume file":
    preferences = load_sample_resume()

elif mode == "Fill out form":
    st.subheader("Enter Your Job Preferences")
    values = st.multiselect("Work Values", ["Impactful Work", "Mentorship & Career Development", "Work-Life Balance", "Transparency & Communication"])
    role_types = st.multiselect("Role Types", ["Full-Time", "Contract", "Part-Time"])
    titles = st.text_area("Preferred Job Titles (comma separated)").split(",")
    locations = st.text_area("Preferred Locations (comma separated)").split(",")
    role_level = st.text_input("Role Level (e.g., Senior (5 to 8 years))")
    leadership_preference = st.text_input("Leadership Preference (e.g., Individual Contributor)")
    company_size = st.multiselect("Company Size", ["1-50 Employees", "51-200 Employees", "201-500 Employees", "500+ Employees"])
    industries = st.multiselect("Industries", ["AI & Machine Learning", "Design", "Software", "Finance", "E-commerce", "Automotive", "Media & Entertainment", "Semiconductors"])
    skills = st.text_area("Skills (comma separated)").split(",")
    min_salary = st.number_input("Minimum Salary", min_value=0, value=50000, step=5000)

    preferences = {
        "values": [v.strip() for v in values if v.strip()],
        "role_types": [r.strip() for r in role_types if r.strip()],
        "titles": [t.strip() for t in titles if t.strip()],
        "locations": [l.strip() for l in locations if l.strip()],
        "role_level": [role_level] if role_level else [],
        "leadership_preference": leadership_preference,
        "company_size": [c.strip() for c in company_size if c.strip()],
        "industries": [i.strip() for i in industries if i.strip()],
        "skills": [s.strip() for s in skills if s.strip()],
        "min_salary": min_salary
    }

if st.button("Run Recommendations"):
    results = []
    for job in jobs:
        match = compute_match(job, preferences)
        results.append({
            "job_id": job["job_id"],
            "job_title": job["title"],
            "match_score": match
        })

    # Sort and filter
    sorted_results = sorted(results, key=lambda x: x["match_score"], reverse=True)
    threshold = 30  # Lowered so more jobs show up
    filtered_results = [r for r in sorted_results if r["match_score"] >= threshold]

    st.subheader("Recommended Jobs")
    for r in filtered_results:
        st.write(f"{r['job_id']} — {r['job_title']} — **{r['match_score']}%**")
