# ライブラリのインポート
import subprocess
subprocess.run(["pip", "install", "scikit-learn"])

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
     "過去2週間、ほとんど毎日気分が沈んでいたか？",
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

# Streamlit UI
st.title("メンタルヘルス分析アプリ")

# 画像入力
img_source = st.radio("画像のソースを選択してください。", ("画像をアップロード", "カメラで撮影"))
if img_source == "カメラで撮影":
    img_file_buffer = st.camera_input("カメラで撮影")
elif img_source == "画像をアップロード":
    img_file_buffer = st.file_uploader("ファイルを選択")
else:
    img_file_buffer = None

# 画像の感情分析
if img_file_buffer:
    img_file_buffer_2 = Image.open(img_file_buffer)
    img_file = np.array(img_file_buffer_2)
    cv2.imwrite('temporary.jpg', img_file)
    image_prediction = detector.detect_image("temporary.jpg")
    image_prediction = image_prediction[["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]]
    emotion = image_prediction.idxmax(axis=1)[0]
    st.markdown("#### あなたの表情は")
    st.markdown("### {}です".format(emotion))

# PHQ-9アンケート
st.subheader("PHQ-9アンケート")
phq_answers = []
for q in phq_questions:
    answer = st.selectbox(q, phq_choices)
    phq_answers.append(phq_choices.index(answer))
phq_9_score = sum(phq_answers)

# データ組み合わせ
if img_file_buffer:
    data = image_prediction.iloc[0].to_dict()
else:
    data = {}
data['phq_9_score'] = phq_9_score
df = pd.DataFrame([data])

# ランダムフォレスト（ここではダミーコード）
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

# 予測（ダミーコード）
# prediction = clf.predict(df)

# 結果表示（ダミーコード）
# st.subheader(f"予測結果: {prediction}")
