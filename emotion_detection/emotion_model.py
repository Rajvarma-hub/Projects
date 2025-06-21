from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()
api_token=os.getenv("api_key")

def emotion(text):
    model = "j-hartmann/emotion-english-distilroberta-base"
    client=InferenceClient(token=st.secrets['api_key'])
    reponse=client.text_classification(text=text,model=model)
    sorted_emotions=max(reponse,key=lambda x:x['score'])
    return sorted_emotions
