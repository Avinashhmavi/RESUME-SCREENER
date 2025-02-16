import PyPDF2
import re
import streamlit as st
import tempfile
import time
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load SentenceTransformer model for local fallback
model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')  # 'mps' for Mac M1/M2 GPU

# 🔹 Replace with your actual API Keys
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # ✅ Fixed URL

OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

HF_API_KEY = "yhf_ceusBQxwYwVBSGHAuGNdQOCmvGmHoVMIuV"
HF_API_URL = "https://api-inference.huggingface.co/models/google/gemini-1.5-pro"

MODEL_NAME = "mixtral-8x7b-32768"  # Default Groq model

def extract_text_from_pdf(uploaded_file):
    """Extracts and cleans text from an uploaded PDF resume."""
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return preprocess_text(text.strip())

def preprocess_text(text):
    """Cleans and preprocesses text by removing special characters and extra spaces."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

def analyze_match(resume_text, job_description):
    """Analyzes resume vs job description using APIs, with fallback to local processing."""
    max_length = 3000  # Truncate long inputs
    truncated_resume = resume_text[:max_length]
    truncated_jd = job_description[:max_length]

    prompt = f"""
    Analyze the match between this resume and job description.
    Consider key skills, experience requirements, and qualifications.
    Respond ONLY with a percentage match score between 0-100 without any additional text.

    Resume: {truncated_resume}
    Job Description: {truncated_jd}
    Match Score: """

    def call_api(api_url, api_key, headers):
        """Helper function to call an API and parse response."""
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 10
            }
            response = requests.post(api_url, headers=headers, json=payload)

            if response.status_code != 200:
                raise Exception(f"API Error: {response.text}")

            response_text = response.json()['choices'][0]['message']['content'].strip()
            match = re.search(r'\b(\d{1,3})\b', response_text)

            if match:
                return max(0, min(100, float(match.group(1))))  # Ensure score is 0-100
        except Exception as e:
            st.warning(f"❌ Failed to use API {api_url}: {e}")
            return None

    # Try APIs in order
    for api_url, api_key in [(GROQ_API_URL, GROQ_API_KEY), (OPENROUTER_API_URL, OPENROUTER_API_KEY), (HF_API_URL, HF_API_KEY)]:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        match_score = call_api(api_url, api_key, headers)
        if match_score is not None:
            return match_score

    # Final Fallback: Local SentenceTransformer model
    st.warning("🚨 All APIs failed. Using local model for analysis.")
    embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
    cosine_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return cosine_similarity * 100

def generate_resume_improvement_suggestions(resume_text, job_description):
    """Generates **detailed** resume improvement suggestions using AI."""
    prompt = f"""
    You are an expert HR consultant. Analyze this resume based on the job description and provide the following:
    1️⃣ Key **skills missing** from the resume that should be added.
    2️⃣ If the resume is **ATS-friendly** or needs formatting improvements.
    3️⃣ **Additional projects** or work experience that can enhance alignment.
    4️⃣ **Specific phrases** or keywords to improve the resume.

    Resume: {resume_text[:3000]}
    Job Description: {job_description[:3000]}
    """

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    try:
        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 300}
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "⚠️ Failed to generate detailed recommendations. Try refining your resume manually."
    except Exception as e:
        return f"⚠️ Error generating resume recommendations: {e}"

# Streamlit UI Setup
st.set_page_config(page_title="AI Resume & Job Matcher", page_icon="📄", layout="centered")
st.image("resume.jpeg", use_column_width=True)

st.title("📄 AI Resume & Job Matcher")
st.write("### Upload your resume & job description to see how well they match!")

# File Upload
uploaded_file = st.file_uploader("📂 Upload your resume (PDF)", type=["pdf"])
resume_text = ""

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        resume_text = extract_text_from_pdf(temp_file.name)
    st.success("✅ Resume uploaded successfully!")

# Job Description Input
job_description = st.text_area("📝 Paste the job description here:", height=200, placeholder="Enter the job description...")

if st.button("Analyze Match ✅"):
    if resume_text and job_description:
        with st.spinner("🔍 Analyzing the match... Please wait."):
            time.sleep(2)
            processed_resume = preprocess_text(resume_text)
            processed_job = preprocess_text(job_description)
            
            try:
                match_score = analyze_match(processed_resume, processed_job)
                recommendations = generate_resume_improvement_suggestions(processed_resume, processed_job)
            except Exception as e:
                st.error(f"Failed to analyze match: {e}")
                st.stop()
        
        st.success(f"🔥 Match Score: {match_score:.2f}%")
        st.progress(match_score / 100)
        
        if match_score > 80:
            st.balloons()
            st.write("🎉 Excellent match! Your resume is highly aligned with this job.")
        elif match_score > 50:
            st.write("👍 Good match! Consider fine-tuning your resume for a better fit.")
        else:
            st.write("⚠️ Weak match. Modify your resume for better alignment.")
        
        st.write("### 📊 Resume Analysis Insights")
        labels = ["Match", "Missing Skills"]
        values = [match_score, 100 - match_score]
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#FF5252'], startangle=140)
        st.pyplot(fig)

        st.write("### 📌 **Detailed Resume Recommendations**")
        st.write(recommendations)
    else:
        st.warning("⚠️ Please upload a resume and enter a job description!")