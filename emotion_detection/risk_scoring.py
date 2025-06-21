emotion_risk_map={
    'anger':5,
    'disgust':4,
    'fear':3,
    "sadness":3,
    "joy":1,
    "love":0

}

def get_risk_score(emotion):
    return emotion_risk_map.get(emotion.lower(),0)