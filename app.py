import PyPDF2
import re
import streamlit as st
import tempfile
import time
import requests
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

device = "cpu"  # Force CPU mode for compatibility
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

print("✅ Using CPU mode (Streamlit Cloud safe)")
# 🔹 Replace with your actual API Keys
import streamlit as st
import requests

# ✅ Get API keys securely from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_URL = st.secrets["GROQ_API_URL"]

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_API_URL = st.secrets["OPENROUTER_API_URL"]

HF_API_KEY = st.secrets["HF_API_KEY"]
HF_API_URL = st.secrets["HF_API_URL"]

MODEL_NAME = "llama3-70b-8192"  # Default Groq model

# Configuration Constants
MAX_TEXT_LENGTH = 3000  # Maximum length for text inputs to LLMs to prevent overly long prompts
DEFAULT_MODEL_NAME = "llama3-70b-8192"  # Default model for API calls, matches MODEL_NAME
API_CALL_TEMPERATURE = 0.2  # Creativity/randomness for match analysis (lower is more deterministic)
API_CALL_MAX_TOKENS_MATCH = 10  # Max tokens for match score response (should be small for just a percentage)
API_CALL_TEMPERATURE_SUGGESTIONS = 0.3  # Creativity for improvement suggestions
API_CALL_MAX_TOKENS_SUGGESTIONS = 300 # Max tokens for suggestion generation

def extract_text_from_pdf(file_object):
    """
    Extracts and cleans text from an uploaded PDF file object.
    Note: Expects a file object (e.g., from tempfile or BytesIO) not a path.
    """
    text = ""
    pdf_reader = PyPDF2.PdfReader(file_object)  # PyPDF2 reads directly from the file object
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return preprocess_text(text.strip())

def preprocess_text(text):
    """Cleans and preprocesses text by removing special characters and extra spaces."""
    text = text.lower().strip()  # Convert to lowercase and remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace characters with a single space
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove any character that is not a letter, number, or whitespace
    return text

def analyze_match(resume_text, job_description):
    """Analyzes resume vs job description using APIs, with fallback to local processing."""
    # Truncate texts to avoid exceeding API limits and keep prompts manageable
    truncated_resume = resume_text[:MAX_TEXT_LENGTH]
    truncated_jd = job_description[:MAX_TEXT_LENGTH]

    # LLM prompt designed to elicit a direct percentage score
    # It specifies the task, context (resume vs. JD), and desired output format.
    prompt = f"""
    Analyze the match between this resume and job description.
    Consider key skills, experience requirements, and qualifications.
    Respond ONLY with a percentage match score between 0-100 without any additional text.

    Resume: {truncated_resume}
    Job Description: {truncated_jd}
    Match Score: """

    def call_api(api_url, api_key, headers):
        """
        Helper function to make a generic API call to an LLM.
        Parses the response to extract a match score.
        """
        try:
            payload = {
                "model": MODEL_NAME,  # Using existing global MODEL_NAME
                "messages": [{"role": "user", "content": prompt}],
                "temperature": API_CALL_TEMPERATURE,
                "max_tokens": API_CALL_MAX_TOKENS_MATCH
            }
            response = requests.post(api_url, headers=headers, json=payload)

            if response.status_code != 200:
                raise Exception(f"API Error: {response.text}")

            response_text = response.json()['choices'][0]['message']['content'].strip()
            # Regex to find a 1 to 3 digit number (the percentage score)
            match = re.search(r'\b(\d{1,3})\b', response_text)

            if match:
                # Convert extracted score to float, clamp between 0 and 100
                return max(0, min(100, float(match.group(1))))
        except Exception as e:
            st.warning(f"❌ Failed to use API {api_url}: {e}")
            return None

    # Attempt to get match score from APIs in a preferred order: Groq, OpenRouter, HuggingFace
    for api_url, api_key in [(GROQ_API_URL, GROQ_API_KEY), (OPENROUTER_API_URL, OPENROUTER_API_KEY), (HF_API_URL, HF_API_KEY)]:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        match_score = call_api(api_url, api_key, headers)
        if match_score is not None:
            return match_score

    # Fallback to local SentenceTransformer model if all API calls fail
    st.warning("🚨 All APIs failed. Using local model for analysis.")
    embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
    # Calculate cosine similarity and scale to a percentage
    cosine_similarity = sentence_transformers.util.cos_sim(embeddings[0], embeddings[1]).item()
    return cosine_similarity * 100

def generate_resume_improvement_suggestions(resume_text, job_description):
    """Generates **detailed** resume improvement suggestions using AI."""
    # Prompt for LLM, instructing it to act as an HR consultant and provide specific feedback categories.
    prompt = f"""
    You are an expert HR consultant. Analyze this resume based on the job description and provide the following:
    1️⃣ Key **skills missing** from the resume that should be added.
    2️⃣ If the resume is **ATS-friendly** or needs formatting improvements.
    3️⃣ **Additional projects** or work experience that can enhance alignment.
    4️⃣ **Specific phrases** or keywords to improve the resume.

    Resume: {resume_text[:MAX_TEXT_LENGTH]}
    Job Description: {job_description[:MAX_TEXT_LENGTH]}
    """

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    try:
        # API call specifically to Groq for generating suggestions
        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": API_CALL_TEMPERATURE_SUGGESTIONS, "max_tokens": API_CALL_MAX_TOKENS_SUGGESTIONS}
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
    # Use a temporary file to handle the uploaded PDF.
    # `delete=True` by default, so the file is automatically removed when the context is exited.
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file_obj:
        temp_file_obj.write(uploaded_file.read())
        temp_file_obj.seek(0) # Reset file pointer to the beginning for reading
        resume_text = extract_text_from_pdf(temp_file_obj) # Pass the file object directly
    st.success("✅ Resume uploaded successfully!")

# Job Description Input
job_description = st.text_area("📝 Paste the job description here:", height=200, placeholder="Enter the job description...")

if st.button("Analyze Match ✅"):
    if resume_text and job_description:
        with st.spinner("🔍 Analyzing the match... Please wait."):
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
        
        st.write("### 📌 **Detailed Resume Recommendations**")
        st.write(recommendations)
    else:
        st.warning("⚠️ Please upload a resume and enter a job description!")
