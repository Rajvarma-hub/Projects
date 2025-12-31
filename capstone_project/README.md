# 🤖 AI Resume Tailoring & Cover Letter Generator

An AI-powered application that automatically **tailors resumes**, **scores ATS compatibility**, and **generates customized cover letters** based on a given job description.

The system uses **Gemini LLM (serverless inference)** to optimize resumes for Applicant Tracking Systems (ATS), helping candidates significantly improve their shortlisting chances.

---

## 🚀 Features

- 📄 Upload resume in **PDF or DOCX** format  
- 📝 Paste any job description  
- 🧠 AI-powered **resume tailoring** with keyword optimization  
- 📊 **ATS compatibility scoring** (out of 100) with improvement suggestions  
- ✉️ **Customized cover letter generation**  
- ✅ Output verification using an AI feedback loop  
- 📥 Download final outputs in **PDF & DOCX formats**  
- 🖥️ Clean and interactive **Streamlit UI**

---

## 🧠 How It Works

1. User uploads a resume (PDF/DOCX)
2. Job description is provided
3. Resume text is extracted and cleaned
4. Gemini LLM performs:
   - Resume tailoring
   - ATS scoring & feedback
   - Cover letter generation
5. AI-based verification ensures output quality
6. Final documents are available for download

---

## 🧩 System Architecture

Resume (PDF/DOCX)  
→ Text Extraction & Cleaning  
→ Gemini LLM (Resume Tailoring)  
→ ATS Scoring Engine  
→ Cover Letter Generator  
→ Verification Loop  
→ Downloadable Resume & Cover Letter

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Frontend:** Streamlit  
- **LLM:** Google Gemini (`gemini-2.5-flash`)  
- **AI Integration:** `google-genai` SDK  
- **File Handling:** PyPDF2, python-docx, FPDF  
- **Environment Management:** python-dotenv  
- **Animations:** streamlit-lottie  

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository

git clone https://github.com/Rajvarma-hub/AI-Resume-Generator.git
cd AI-Resume-Generator
2️⃣ Create and configure .env
env
Copy code
GEMINI_API_KEY=your_api_key_here
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the application
bash
Copy code
streamlit run app.py
📁 Project Structure
bash
Copy code
AI-Resume-Generator/
├── app.py                     # Main Streamlit application
├── requirements.txt
├── .env                       # API key configuration
├── README.md
💡 Use Cases
Job seekers optimizing resumes for ATS

Freshers and professionals applying to multiple roles

Career platforms offering resume optimization

HR-tech and recruitment automation tools

🔮 Future Enhancements
Multi-job comparison support

Resume version history

LinkedIn profile optimization

Multi-language resume support

API-based integration for job portals

👤 Author
Raj
AI Engineer | Backend Developer

GitHub: https://github.com/Rajvarma-hub

LinkedIn: https://linkedin.com/in/your-profile

