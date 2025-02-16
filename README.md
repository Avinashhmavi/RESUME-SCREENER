# 📄🔍 AI Resume & Job Matcher

Welcome to **AI Resume & Job Matcher**, a powerful tool designed to help job seekers analyze how well their resume aligns with a specific job description. Using advanced AI models and natural language processing (NLP), this app provides detailed insights, actionable recommendations, and a match score to enhance your chances of landing your dream job!


# Resume Screener App  

This project includes a **Resume Screener** tool. You can access the live application here:  

[👉 Click Here to Open Resume Screener](https://resume-screener-ifim.streamlit.app/)  

---

## 🌟 Features

- **Resume Upload**: Seamlessly upload your PDF resume for analysis.
- **Job Description Input**: Paste the job description text directly into the app.
- **Match Score Analysis**: Get a percentage match score between your resume and the job description.
- **Detailed Recommendations**:
  - Identify **missing skills** from your resume.
  - Check if your resume is **ATS-friendly**.
  - Suggest **additional projects** or work experience to improve alignment.
  - Provide **specific keywords/phrases** to optimize your resume.
- **Visual Insights**: View a pie chart summarizing the match score and missing skills.
- **Fallback Mechanism**: If external APIs fail, the app uses a local SentenceTransformer model for analysis.
- **Interactive UI**: A clean, user-friendly interface powered by Streamlit.

---

## 🔧 Pre-installation Requirements

Before running the app, ensure you have the following installed:

1. **Python 3.8+**
2. **Streamlit**: `pip install streamlit`
3. **PyPDF2**: `pip install PyPDF2`
4. **Sentence Transformers**: `pip install sentence-transformers`
5. **Requests**: `pip install requests`
6. **Matplotlib & Seaborn**: `pip install matplotlib seaborn`
7. **NumPy & Torch**: `pip install numpy torch`

### Optional Dependencies
- **Streamlit Secrets**: Ensure you have API keys stored securely in Streamlit's `secrets.toml` file.

---

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Avinashhmavi/RESUME-SCREENER.git
   cd ai-resume-job-matcher
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - Create a `secrets.toml` file in the `.streamlit` directory:
     ```toml
     GROQ_API_KEY = "your_groq_api_key"
     GROQ_API_URL = "https://api.groq.com/v1"
     OPENROUTER_API_KEY = "your_openrouter_api_key"
     OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
     HF_API_KEY = "your_huggingface_api_key"
     HF_API_URL = "https://api-inference.huggingface.co/models"
     ```
   - Replace placeholders with your actual API keys.

4. Run the app:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`.

---

## 🎨 How It Works

1. **Upload Your Resume**:
   - Drag and drop your PDF resume or click to upload.
   - The app extracts and preprocesses the text.

2. **Paste Job Description**:
   - Copy and paste the job description into the provided text box.

3. **Analyze Match**:
   - Click the "Analyze Match ✅" button.
   - The app calculates a match score and generates detailed recommendations.

4. **View Results**:
   - See your match score as a percentage.
   - Visualize insights with a pie chart.
   - Read actionable suggestions to improve your resume.

---

## 📊 Example Output

### Match Score
- **Excellent Match (>80%)**: Celebrate with balloons! 🎉
- **Good Match (50%-80%)**: Fine-tune your resume for better results. 👍
- **Weak Match (<50%)**: Consider significant updates to align better. ⚠️

### Pie Chart
A colorful pie chart shows the breakdown of your match score vs. missing skills.

### Detailed Recommendations
- **Missing Skills**: Add these to your resume to stand out.
- **ATS-Friendly Check**: Ensure your resume passes automated screening systems.
- **Additional Projects**: Highlight relevant projects to boost alignment.
- **Keywords/Phrases**: Incorporate these to make your resume more impactful.

---

## 🛠️ Technologies Used

- **Streamlit**: For building the interactive web app.
- **PyPDF2**: To extract text from PDF resumes.
- **Sentence Transformers**: For generating embeddings and fallback similarity analysis.
- **Matplotlib & Seaborn**: For visualizing match scores.
- **External APIs**:
  - Groq API
  - OpenRouter API
  - Hugging Face Inference API

---

## 📝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---



## 🙏 Acknowledgments

- Thanks to the creators of **Streamlit**, **Sentence Transformers**, and other libraries for making this project possible.
- Special thanks to the AI community for open-sourcing models and tools that empower developers worldwide.

---



🌟 **Happy Job Hunting!** 🌟
