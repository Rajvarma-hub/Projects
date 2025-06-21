import smtplib
import os
from dotenv import load_dotenv
from email.message import EmailMessage

load_dotenv()
def send_email_alert(message,emotion,risk):
    password=os.getenv('password')
    msg=EmailMessage()
    msg['Subject']='Customer Frustration Alert'
    msg['From']='learnershub124@gmail.com'
    msg['To']='rajkumarthirthala2005@gmail.com'
    msg.set_content(f"""
                    Frustration Detected !
                    Message:"{message}"
                    Emotion: "{emotion}"
                    Risk: "{risk}"
                    
                    Action Recommended: Escalate immediately.
                    """)
    with smtplib.SMTP('smtp.gmail.com',587) as stmp:
        stmp.starttls()
        stmp.login('learnershub124@gmail.com',password)
        stmp.send_message(msg)