# 🧠 EmotionGuard – Real-Time Customer Frustration Detection System

EmotionGuard is a real-time AI-powered application that analyzes customer messages to detect emotional sentiment, assess escalation risk, and automatically alert support teams when critical intervention is required.

The system helps businesses reduce customer churn, improve support response time, and proactively manage high-risk customer interactions.

---

## 🚀 Features

- 🔍 Real-time emotion detection using NLP
- 📊 Confidence scoring for emotional intensity
- ⚠️ Risk-based escalation logic
- 📧 Automated email alerts for high-risk messages
- 🧠 Serverless inference (no model training or hosting)
- 🖥️ Interactive Streamlit dashboard

---

## 🧠 How It Works

1. User enters a customer message in the Streamlit UI  
2. The message is sent to Hugging Face’s serverless Inference API  
3. The most probable emotion and confidence score are extracted  
4. A risk score is calculated using custom emotion-to-risk mapping  
5. High-risk emotions automatically trigger an email alert  

---

## 🧩 System Architecture

Customer Message  
→ Emotion Detection (Hugging Face Inference API)  
→ Emotion Label + Confidence Score  
→ Risk Scoring Engine  
→ Low / Medium / High Risk Classification  
→ Email Escalation (High Risk)

---

## 🧠 Emotion-to-Risk Mapping

| Emotion   | Risk Score |
|---------|------------|
| Anger   | 5 |
| Disgust | 4 |
| Fear    | 3 |
| Sadness | 3 |
| Joy     | 1 |
| Love    | 0 |

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Frontend:** Streamlit  
- **Emotion Analysis:** Hugging Face Inference API  
- **Model Used:** `j-hartmann/emotion-english-distilroberta-base`  
- **Backend Logic:** Custom risk scoring engine  
- **Email Alerts:** `smtplib`, `email.message`, `dotenv`

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
 bash
git clone  https://github.com/Rajvarma-hub/Projects/edit/main/emotion_detection
cd EmotionGuard

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
streamlit run app.py

📁 Project Structure
EmotionGuard/
├── app.py                 # Streamlit application
├── emotion_model.py       # Emotion detection logic
├── risk_scoring.py        # Emotion-to-risk mapping
├── email_alert.py         # Email notification module
├── requirements.txt
└── README.md

💡 Use Cases

Customer support escalation systems

Call center monitoring tools

SaaS customer retention platforms

AI-powered CRM enhancements

🔮 Future Enhancements

Emotion trend analysis across conversations

CRM integrations (Zendesk, Freshdesk, Salesforce)

Multilingual emotion detection

REST API & webhook support

Dashboard analytics for support teams

👤 Author

Raj
AI Engineer | Backend Developer

GitHub: https://github.com/Rajvarma-hub

LinkedIn: https://linkedin.com/in/your-profile
