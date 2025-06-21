import streamlit as st
from email_alert import send_email_alert
from emotion_model import emotion
from risk_scoring import get_risk_score
st.set_page_config(page_title="EmotionGuad",page_icon="🧠")
st.title("EmotionGuard: Customer Frustration Detector")

msg=st.text_area(" 🛡️ Enter the Customer Message")
if st.button("🔍 Analyze Emotion"):
    if not msg.strip():
        st.warning("Please enter a message")
    else:
        with st.spinner("Detecting Emotion...."):
            result=emotion(msg)
            label=result['label']
            score=round(result['score']*100,2)
            risk=get_risk_score(label)
        st.markdown(f"### 🔥 Top Emotion: `{label.upper()}` ({score}%)")
        st.progress(score/100)
        if risk>=4:
            st.error("🚨 HIGH RISK: Escalation required!")
            send_email_alert(msg,label,score/100)
            print("Email sent")
        elif risk >= 2:
            st.warning("⚠️ Medium Risk: Monitor carefully.")
        else:
            st.success("✅ Low Risk: No action needed.")
          