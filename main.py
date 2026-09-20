import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import engine, get_db, Base
import models
from schemas import PredictRequest, PredictResponse

# สร้างตารางในฐานข้อมูลอัตโนมัติ (ถ้ายังไม่มี) ตอนแอปเริ่มทำงาน
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # โปรดักชันจริงควรระบุ origin ของหน้าเว็บให้ชัดเจน
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลด "artifact" ที่ train.py เซฟไว้ (เป็น dict ห่อโมเดล + ข้อมูลสำหรับคำนวณ index)
# ห้ามใช้ pickle.load ตรงๆ เพราะโมเดลถูกเซฟด้วย joblib.dump
artifact = joblib.load("model.pkl")

MODEL = artifact["model"]
FEATURE_ORDER = artifact["feature_order"]
LABELS = artifact["labels"]                 # {0:"low",1:"medium",2:"high",3:"highest"}
POS_COLS = artifact["index_pos_cols"]
NEG_COLS = artifact["index_neg_cols"]
COL_MEANS = artifact["index_col_means"]
COL_STDS = artifact["index_col_stds"]
RAW_MIN = artifact["raw_index_min"]
RAW_MAX = artifact["raw_index_max"]

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


def compute_stress_index(payload: dict) -> float:
    """คำนวณ stress_index (0-100) ด้วยสูตรเดียวกับตอนเทรน (z-score POS - NEG)"""
    total = 0.0
    for col in POS_COLS:
        total += (payload[col] - COL_MEANS[col]) / COL_STDS[col]
    for col in NEG_COLS:
        total -= (payload[col] - COL_MEANS[col]) / COL_STDS[col]
    scaled = (total - RAW_MIN) / (RAW_MAX - RAW_MIN) * 100
    return float(np.clip(scaled, 0, 100))


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Stress Scouter API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, db: Session = Depends(get_db)):
    data = payload.dict()

    # เรียงคอลัมน์ให้ตรงกับตอนเทรนเป๊ะๆ ก่อนส่งเข้าโมเดล
    row = pd.DataFrame([[data[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    pred_class = int(MODEL.predict(row)[0])
    level_key = LABELS[pred_class]
    info = LEVEL_INFO[level_key]

    score = round(compute_stress_index(data))  # 0-100 แบบละเอียด เก็บลง DB (ไม่ส่งกลับให้ผู้ใช้เห็น)

    # บันทึกผลลง database ก่อน ยังไม่ตั้ง respondent_code (ต้องรู้ id ที่ database gen ให้ก่อน)
    result = models.StressResult(
        respondent_code="",   # ใส่ชั่วคราว จะอัปเดตด้านล่างหลังรู้ id จริง
        score=score,
        level=level_key,
    )
    db.add(result)
    db.flush()  # ให้ database gen id มาก่อน โดยยังไม่ commit

    result.respondent_code = f"{result.id:05d}"  # คนแรก 00001, คนสอง 00002, ...
    db.commit()

    # ส่งกลับให้หน้าเว็บ - ไม่มี score ติดไปด้วย บอกแค่ระดับ
    return PredictResponse(
        label=info["th"],
        level_key=level_key,
        color=info["color"],
        description=info["description"],
    )
