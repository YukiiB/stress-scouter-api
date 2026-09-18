"""
Stress Scouter - Backend API
โหลด model.pkl (จาก train.py) มาเสิร์ฟ endpoint POST /predict
รัน local: uvicorn main:app --reload
Deploy: Render / Railway (ดู requirements.txt คู่กัน)
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Stress Scouter API")

# อนุญาตให้หน้าเว็บ (ไม่ว่าจะโฮสต์ที่ไหน เช่น Google Sites) เรียก API นี้ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # โปรดักชันจริงแนะนำระบุโดเมนเว็บของคุณแทน "*"
    allow_methods=["POST"],
    allow_headers=["*"],
)

ARTIFACT_PATH = "model.pkl"
artifact = joblib.load(ARTIFACT_PATH)

MODEL = artifact["model"]
FEATURE_ORDER = artifact["feature_order"]
LABELS = artifact["labels"]                 # {0:"low",1:"medium",2:"high",3:"highest"}
POS_COLS = artifact["index_pos_cols"]
NEG_COLS = artifact["index_neg_cols"]
COL_MEANS = artifact["index_col_means"]
COL_STDS = artifact["index_col_stds"]
RAW_MIN = artifact["raw_index_min"]
RAW_MAX = artifact["raw_index_max"]

# ข้อความ + สีประจำแต่ละระดับ (แก้ข้อความให้เข้ากับโทนเว็บได้ตามต้องการ)
LEVEL_INFO = {
    "low": {
        "th": "ความเครียดต่ำ",
        "color": "#2e7d32",
        "description": "ตอนนี้คุณจัดการความเครียดได้ดี ลองรักษาสมดุลชีวิตแบบนี้ต่อไป",
    },
    "medium": {
        "th": "ความเครียดปานกลาง",
        "color": "#f9a825",
        "description": "เริ่มมีความเครียดสะสมบ้าง ลองพักผ่อนให้เพียงพอและหาเวลาผ่อนคลายเพิ่มขึ้น",
    },
    "high": {
        "th": "ความเครียดสูง",
        "color": "#ef6c00",
        "description": "ความเครียดอยู่ในระดับสูง ควรหาวิธีจัดการความเครียดอย่างจริงจัง เช่น พูดคุยกับคนที่ไว้ใจหรือผู้เชี่ยวชาญ",
    },
    "highest": {
        "th": "ความเครียดสูงมาก",
        "color": "#c62828",
        "description": "ความเครียดอยู่ในระดับสูงมาก แนะนำให้ปรึกษาผู้เชี่ยวชาญด้านสุขภาพจิตโดยเร็ว",
    },
}


class StressInput(BaseModel):
    anxiety_level: float
    self_esteem: float
    mental_health_history: float
    depression: float
    headache: float
    blood_pressure: float
    sleep_quality: float
    breathing_problem: float
    noise_level: float
    living_conditions: float
    safety: float
    basic_needs: float
    academic_performance: float
    study_load: float
    teacher_student_relationship: float
    future_career_concerns: float
    social_support: float
    peer_pressure: float
    extracurricular_activities: float
    bullying: float


def compute_stress_index(payload: dict) -> float:
    """คำนวณ stress_index (0-100) ด้วยสูตรเดียวกับตอนเทรน (z-score POS - NEG)"""
    total = 0.0
    for col in POS_COLS:
        z = (payload[col] - COL_MEANS[col]) / COL_STDS[col]
        total += z
    for col in NEG_COLS:
        z = (payload[col] - COL_MEANS[col]) / COL_STDS[col]
        total -= z
    scaled = (total - RAW_MIN) / (RAW_MAX - RAW_MIN) * 100
    return float(np.clip(scaled, 0, 100))


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Stress Scouter API is running"}


@app.post("/predict")
def predict(data: StressInput):
    payload = data.dict()

    missing = [c for c in FEATURE_ORDER if c not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"ขาดฟีเจอร์: {missing}")

    # เรียงคอลัมน์ให้ตรงกับตอนเทรนเป๊ะๆ ก่อนส่งเข้าโมเดล
    row = pd.DataFrame([[payload[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

    pred_class = int(MODEL.predict(row)[0])
    label_en = LABELS[pred_class]
    info = LEVEL_INFO[label_en]

    score = compute_stress_index(payload)

    # ความน่าจะเป็นของแต่ละระดับ (ถ้าโมเดลรองรับ predict_proba)
    proba = None
    if hasattr(MODEL, "predict_proba"):
        proba_arr = MODEL.predict_proba(row)[0]
        proba = {LABELS[i]: round(float(p), 3) for i, p in enumerate(proba_arr)}

    return {
        "label": info["th"],
        "level_key": label_en,
        "color": info["color"],
        "description": info["description"],
        "score": round(score, 1),
        "probabilities": proba,
    }
