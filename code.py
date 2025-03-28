# ライブラリのインポート
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from feat import Detector

# py-featの設定
face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "svm"
emotion_model = "resmasknet"
detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model, emotion_model=emotion_model)

# PHQ-9の質問
phq_questions = [
     "過去2週間、ほとんど毎日気分が沈んでいたか？?",
        "過去2週間、興味や喜びを感じることが減ったか？",
        "過去2週間、睡眠障害（寝すぎまたは不眠）があったか？",
        "過去2週間、疲労やエネルギーの低下を感じたか？",
        "過去2週間、食欲の変化（過食または食欲不振）があったか？",
        "過去2週間、自分自身を価値がないと感じたか？",
        "過去2週間、集中力の低下や決断力の低下を感じたか？",
        "過去2週間、動きが遅くなった、または落ち着きがなくなったと感じられたか？",
        "過去2週間、自分を傷つけることや自殺を考えたか？"
    ]
phq_choices = ["全くない", "いくつかの日", "週の半分以上", "ほとんど毎日"]

st.title("メンタルヘルス分析アプリ")



# PHQ-9アンケート
st.subheader("PHQ-9アンケート")
phq_answers = []
for q in phq_questions:
    answer = st.selectbox(q, phq_choices)
    phq_answers.append(phq_choices.index(answer))
phq_9_score = sum(phq_answers)


