# 🛡️ EmotiGuard: Real-Time Customer Frustration Detector

**EmotiGuard** is a real-time AI-powered application that detects **customer frustration** in support chats using emotion analysis. It uses **Hugging Face’s serverless Inference API** to analyze emotions and automatically sends alerts for critical cases (like *anger*), helping support teams act quickly and prevent customer churn.

---

## 🚀 Features

- 🔍 Real-time **Emotion Detection** using Hugging Face's Inference API
- 📈 Risk-based alert system using custom emotion-to-risk mapping
- 📧 **Email Alert System** for high-risk messages (e.g., anger)
- 🧠 Built using **serverless inference** (no model training or hosting needed)
- 🖥️ Clean user interface using **Streamlit**

---

## 💡 How It Works

1. User enters a customer message.
2. Message is sent to Hugging Face's `j-hartmann/emotion-english-distilroberta-base` via the `InferenceClient`.
3. The app extracts the most probable emotion.
4. If the emotion is high-risk (e.g., `anger`, `disgust`), it triggers an **email alert** to the support lead.

---

## 🧠 Emotion-to-Risk Mapping

| Emotion   | Risk Score |
|-----------|------------|
| Anger     | 5          |
| Disgust   | 4          |
| Fear      | 3          |
| Sadness   | 3          |
| Joy       | 1          |
| Love      | 0          |

---

## 🛠️ Tech Stack

| Component       | Tool / Library                         |
|------------------|-----------------------------------------|
| Emotion Analysis | Hugging Face `InferenceClient` (Cloud) |
| Model Used       | `j-hartmann/emotion-english-distilroberta-base` |
| Frontend         | Streamlit                              |
| Email Alerts     | Python `smtplib`, `email.message`, `dotenv` |

---

## 📦 Project Structure

