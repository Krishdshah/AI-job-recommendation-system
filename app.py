import json
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re

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
st.title("SkillSync :- AI Job Recommendation System - Team NeoTech ")
preferences = {}
input_method = st.radio(
    "Choose input method",
    ("Upload Resume", "Fill out form")
)


if input_method == "Upload Resume":
    uploaded_resume = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])
    if uploaded_resume:
        import pdfplumber, re
        with pdfplumber.open(uploaded_resume) as pdf:
            resume_text = ""
            for page in pdf.pages:
                resume_text += page.extract_text() + "\n"

        # Extract basic details
        skills_keywords = ["Python","Java","SQL","React","Machine Learning","AWS","Excel",
                          "Flask","Django","TensorFlow","NLP","Power BI","JavaScript","C++"]
        candidate_skills = [
            s for s in skills_keywords
            if re.search(r"\b" + re.escape(s) + r"\b", resume_text, re.I)
        ]

        candidate_location = ""
        for job in jobs:
            if re.search(job["location"], resume_text, re.I):
                candidate_location = job["location"]
                break

        candidate_title = ""
        for job in jobs:
            if re.search(job["title"], resume_text, re.I):
                candidate_title = job["title"]
                break

        st.success("✅ Resume processed successfully!")

        # --- Editable fields (prefilled with extracted data) ---
        st.subheader("Review & Modify Extracted Preferences")
        values = st.multiselect("Work Values",
                                ["Impactful Work", "Mentorship & Career Development",
                                 "Work-Life Balance", "Transparency & Communication"],
                                default=["Impactful Work"]) # Default value added
        role_types = st.multiselect("Role Types", ["Full-Time", "Contract", "Part-Time"],
                                    default=["Full-Time"]) # Default value added
        titles = st.text_area("Preferred Job Titles (comma separated)",
                              value=candidate_title).split(",")
        locations = st.text_area("Preferred Locations (comma separated)",
                                 value=candidate_location).split(",")
        role_level = st.text_input("Role Level (e.g., Senior (5 to 8 years))",
                                   value="Junior (0 to 2 years)") # Default value added
        leadership_preference = st.text_input("Leadership Preference (e.g., Individual Contributor)",
                                              value="Individual Contributor") # Default value added
        company_size = st.multiselect("Company Size",
                                      ["1-50 Employees", "51-200 Employees",
                                       "201-500 Employees", "500+ Employees"],
                                      default=["51-200 Employees", "201-500 Employees"]) # Default value added
        industries = st.multiselect("Industries",
                                    ["AI & Machine Learning", "Design", "Software", "Finance",
                                     "E-commerce", "Automotive", "Media & Entertainment",
                                     "Semiconductors"],
                                    default=["Software"]) # Default value added
        skills = st.text_area("Skills (comma separated)",
                              value=", ".join(candidate_skills)).split(",")
        min_salary = st.number_input("Minimum Salary", min_value=0, value=50000, step=5000)

        # ✅ Final preferences dict
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



elif input_method == "Fill out form":
    st.subheader("Enter Your Job Preferences")
    values = st.multiselect("Work Values", ["Impactful Work", "Mentorship & Career Development", "Work-Life Balance", "Transparency & Communication"],
                            default=["Impactful Work", "Work-Life Balance"]) # Default value added
    role_types = st.multiselect("Role Types", ["Full-Time", "Contract", "Part-Time"],
                                default=["Full-Time"]) # Default value added
    titles = st.text_area("Preferred Job Titles (comma separated)",
                          value="Software Engineer, Data Scientist").split(",") # Default value added
    locations = st.text_area("Preferred Locations (comma separated)",
                             value="Remote, San Francisco").split(",") # Default value added
    role_level = st.text_input("Role Level (e.g., Senior (5 to 8 years))",
                               value="Mid-Level (3 to 5 years)") # Default value added
    leadership_preference = st.text_input("Leadership Preference (e.g., Individual Contributor)",
                                          value="Individual Contributor") # Default value added
    company_size = st.multiselect("Company Size", ["1-50 Employees", "51-200 Employees", "201-500 Employees", "500+ Employees"],
                                  default=["51-200 Employees"]) # Default value added
    industries = st.multiselect("Industries", ["AI & Machine Learning", "Design", "Software", "Finance", "E-commerce", "Automotive", "Media & Entertainment", "Semiconductors"],
                                default=["Software", "AI & Machine Learning"]) # Default value added
    skills = st.text_area("Skills (comma separated)",
                          value="Python, SQL, AWS, Machine Learning").split(",") # Default value added
    min_salary = st.number_input("Minimum Salary", min_value=0, value=75000, step=5000) # Default value added

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
            "location": job.get("location", "N/A"),
            "industry": job.get("industry", "N/A"),
            "employment_type": job.get("employment_type", "N/A"),
            "company_size": job.get("company_size", "N/A"),
            "values": job.get("values_promoted", []),
            "salary_range": job.get("salary_range", []),
            "match_score": match
        })

    # Sort and filter
    sorted_results = sorted(results, key=lambda x: x["match_score"], reverse=True)
    threshold = 30
    filtered_results = [r for r in sorted_results if r["match_score"] >= threshold]

    st.subheader("✨ Recommended Jobs for You")

    if not filtered_results:
        st.warning("No jobs found matching your preferences. Try adjusting filters!")
    else:
        for r in filtered_results:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                        **{r['job_title']}** 📌 *{r['location']}* 🏢 *{r['industry']}* 💼 *{r['employment_type']}*
                    """)
                with col2:
                    st.markdown(
                        f"<div style='text-align:center; font-size:18px; "
                        f"background-color:#1E90FF; color:white; padding:8px; "
                        f"border-radius:10px;'>"
                        f"<b>{r['match_score']}%</b><br/>Match</div>",
                        unsafe_allow_html=True
                    )

                # Expandable section for more details
                with st.expander("🔎 View More Details"):
                    st.write(f"**Job ID:** {r['job_id']}")
                    st.write(f"**Company Size:** {r['company_size']}")
                    st.write(f"**Salary Range:** {r['salary_range'] if r['salary_range'] else 'Not specified'}")
                    st.write(f"**Values Promoted:** {', '.join(r['values']) if r['values'] else 'Not specified'}")

                st.markdown("---")