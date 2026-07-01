import streamlit as st
import pickle
import pandas as pd
import pdfplumber
import re
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
logo = Image.open("ml logo.png")


# ================= LOAD SKILLS =================
skills_df = pd.read_csv("skills.csv", header=None)
SKILLS = skills_df[0].str.lower().tolist()

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="TalentLens AI",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================= PREMIUM SAAS CSS =================
st.markdown("""
<style>

/* Main Background */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top right, rgba(59,130,246,0.35), transparent 30%),
        radial-gradient(circle at bottom left, rgba(6,182,212,0.25), transparent 30%),
        linear-gradient(135deg, #0f172a, #1e3a8a) !important;
    color: white;
}

/* Remove header */
header, [data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    visibility: visible !important;
}

/* Optional: make toolbar blend with background */
[data-testid="stToolbar"] > div {
    background: transparent !important;
}
/* Typography */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: white !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(17, 24, 39, 0.90);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.12);
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 3em;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.15);
    font-weight: bold;
    color: white;
    background: rgba(37,99,235,0.45);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(6,182,212,0.35);
}

/* Uploaders + radio */
[data-testid="stFileUploader"],
[data-testid="stRadio"] {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding: 12px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.12);
}

/* Text Area */
textarea {
    border-radius: 12px !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
}

/* Metric Cards */
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    backdrop-filter: blur(14px);
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 36px rgba(6,182,212,0.35);
}
/* Candidate Cards */
.candidate-card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
col1, col2 = st.columns([1, 5])

with col1:
    st.image(logo, width=120)

with col2:
    st.markdown("""
    <h1 style='color:white; margin-bottom:0;'>
        TalentLens AI
    </h1>
    <h4 style='color:#cbd5e1; font-weight:400; margin-top:0;'>
        AI Resume Analyzer & Hiring Assistant
    </h4>
    """, unsafe_allow_html=True)


# ================= LOAD MODEL =================
category_model = pickle.load(
    open("category_model.pkl", "rb")
)

category_vectorizer = pickle.load(
    open("category_vectorizer.pkl", "rb")
)

# ================= FUNCTIONS =================
def extract_pdf_text(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def extract_txt_text(uploaded_file):
    return uploaded_file.read().decode("utf-8")


def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        if skill in text:
            found.append(skill)

    return found
# ================= SIDEBAR INPUT PANEL =================
with st.sidebar:
    st.image(logo, width=180)
    st.markdown("## TalentLens AI")

    st.markdown("## 📥 Input Panel")

    resume_files = st.file_uploader(
        "Upload Resumes (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    jd_option = st.radio(
        "Job Description Input Method",
        ["Paste Text", "Upload PDF", "Upload TXT"]
    )

    job_description = ""

    if jd_option == "Paste Text":
        job_description = st.text_area(
            "Paste Job Description",
            height=200
        )

    elif jd_option == "Upload PDF":
        jd_file = st.file_uploader(
            "Upload JD PDF",
            type=["pdf"],
            key="jd_pdf"
        )
        if jd_file:
            job_description = extract_pdf_text(jd_file)

    elif jd_option == "Upload TXT":
        jd_file = st.file_uploader(
            "Upload JD TXT",
            type=["txt"],
            key="jd_txt"
        )
        if jd_file:
            job_description = extract_txt_text(jd_file)

    analyze = st.button("Analyze Resume")

# ================= ANALYSIS =================
if analyze:

    if resume_files and job_description:

        results = []

        for resume_file in resume_files:

            if resume_file.type == "application/pdf":
                resume_text = extract_pdf_text(resume_file)
            else:
                resume_text = extract_txt_text(resume_file)

            # Skill Analysis
            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(job_description)

            matched_skills = list(set(resume_skills) & set(jd_skills))
            missing_skills = list(set(jd_skills) - set(resume_skills))

            skill_match_percent = (
                len(matched_skills) / len(jd_skills) * 100
                if jd_skills else 0
            )

            # Category Prediction
            cleaned_resume = clean(resume_text)
            resume_vector = category_vectorizer.transform([cleaned_resume])
            prediction = category_model.predict(resume_vector)[0]

            # ATS Score
            docs = [clean(resume_text), clean(job_description)]
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf = vectorizer.fit_transform(docs)
            score = cosine_similarity(tfidf[0], tfidf[1])[0][0] * 100

            results.append((
                resume_file.name,
                prediction,
                score,
                skill_match_percent,
                missing_skills
            ))

        results = sorted(results, key=lambda x: x[2], reverse=True)

        total_resumes = len(results)
        best_score = round(results[0][2], 2)
        avg_skill = round(sum(r[3] for r in results) / len(results), 2)

        # ================= KPI CARDS =================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>📄 Total Resumes</h4>
                    <h2>{total_resumes}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>🏆 Best ATS Score</h4>
                    <h2>{best_score}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>📊 Avg Skill Match</h4>
                    <h2>{avg_skill}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ================= HR ANALYTICS =================
        st.subheader("📈 HR Analytics Dashboard")

        chart_df = pd.DataFrame(
            results,
            columns=[
                "Resume",
                "Role",
                "Score",
                "Skill Match %",
                "Missing Skills"
            ]
        )

        col1, col2 = st.columns(2)

        with col1:
            fig_score = px.bar(
                chart_df,
                x="Resume",
                y="Score",
                title="ATS Score by Resume",
                text="Score"
            )
            st.plotly_chart(fig_score, use_container_width=True)

        with col2:
            fig_skill = px.bar(
                chart_df,
                x="Resume",
                y="Skill Match %",
                title="Skill Match by Resume",
                text="Skill Match %"
            )
            st.plotly_chart(fig_skill, use_container_width=True)

        role_counts = chart_df["Role"].value_counts().reset_index()
        role_counts.columns = ["Role", "Count"]

        fig_roles = px.pie(
            role_counts,
            names="Role",
            values="Count",
            title="Role Distribution"
        )
        st.plotly_chart(fig_roles, use_container_width=True)

        st.markdown("---")

        # ================= CANDIDATE CARDS =================
        for i, (name, role, score, skill_percent, missing_skills) in enumerate(results, 1):

            st.markdown(
                f"""
                <div class="candidate-card">
                    <h3>#{i} — {name}</h3>
                    <p><b>Predicted Role:</b> {role}</p>
                    <p><b>ATS Score:</b> {round(score, 2)}%</p>
                    <p><b>Skill Match:</b> {round(skill_percent, 2)}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(int(skill_percent))

            if missing_skills:
                st.warning("Missing Skills: " + ", ".join(missing_skills))
            else:
                st.success("✅ No missing skills")

        # ================= TABLE =================
        df_results = pd.DataFrame(
            results,
            columns=[
                "Resume",
                "Role",
                "Score",
                "Skill Match %",
                "Missing Skills"
            ]
        )

        df_results["Score"] = df_results["Score"].round(2)
        df_results["Skill Match %"] = df_results["Skill Match %"].round(2)
        df_results["Missing Skills"] = df_results["Missing Skills"].apply(
            lambda x: ", ".join(x)
        )

        st.subheader("🏆 Resume Ranking")
        st.dataframe(df_results, use_container_width=True)

        # ================= CSV DOWNLOAD =================
        csv = df_results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Ranking CSV",
            data=csv,
            file_name="resume_ranking.csv",
            mime="text/csv"
        )

    else:
        st.warning("Upload resumes and job description first")


