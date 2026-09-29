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

# โหลดไฟล์ที่เทรนไว้ (model.pkl) — เป็น dict ที่เซฟด้วย joblib จาก train.py
# ข้างในมี: model, feature_order, labels, พารามิเตอร์ของ index, index_cuts (เกณฑ์ SPST-20)
artifact = joblib.load("model.pkl")

FEATURE_ORDER = artifact["feature_order"]          # ลำดับคอลัมน์ตอนเทรน
POS_COLS = artifact["index_pos_cols"]
NEG_COLS = artifact["index_neg_cols"]
INDEX_COLS = POS_COLS + NEG_COLS
COL_MEANS = pd.Series(artifact["index_col_means"])
COL_STDS = pd.Series(artifact["index_col_stds"])
RAW_MIN = artifact["raw_index_min"]
RAW_MAX = artifact["raw_index_max"]
CUTS = artifact["index_cuts"]                       # cut-off ตามเกณฑ์ SPST-20 บน index 0-100

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
LABELS = artifact["labels"]  # {0:"low",1:"medium",2:"high",3:"highest"}


def compute_stress_index(row: pd.DataFrame) -> float:
    """z-score ทุก feature -> composite index -> ปรับสเกล 0-100"""
    z = (row[INDEX_COLS] - COL_MEANS) / COL_STDS
    raw = z[POS_COLS].sum(axis=1) - z[NEG_COLS].sum(axis=1)
    idx = (raw - RAW_MIN) / (RAW_MAX - RAW_MIN) * 100
    return float(idx.clip(0, 100).iloc[0])


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Stress Scouter API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, db: Session = Depends(get_db)):
    # 1) เตรียม features ตามลำดับที่โมเดลต้องการ
    row = pd.DataFrame([{name: getattr(payload, name) for name in FEATURE_ORDER}])

    # 2) คำนวณ composite index แล้วแบ่ง 4 ระดับด้วย cut-off ตามเกณฑ์ SPST-20
    stress_index = compute_stress_index(row)
    level_code = int(np.searchsorted(CUTS, stress_index, side="right"))  # 0-3
    level_key = LABELS[level_code]
    info = LEVEL_INFO[level_key]
    score = int(round(stress_index))  # 0-100 เก็บลง DB เท่านั้น ไม่ส่งกลับให้ผู้ใช้เห็น

    # (ทางเลือก) ผลโมเดล 3 ระดับจาก label เดิม ไว้เทียบ/ดีบัก ไม่ได้ใช้กำหนดระดับที่แสดงผล
    # model_level = int(artifact["model"].predict(row)[0])

    # 3) บันทึกผลลง database — ไม่เก็บชื่อ/รหัสนิสิต/คณะ ใช้ respondent_code แทน
    result = models.StressResult(
        respondent_code="",  # ใส่ชั่วคราว จะอัปเดตด้านล่างหลังรู้ id จริง
        score=score,
        level=level_key,
    )
    db.add(result)
    db.flush()  # ให้ database gen id มาก่อน โดยยังไม่ commit

    result.respondent_code = f"{result.id:05d}"  # คนแรก 00001, คนสอง 00002, ...
    db.commit()

    # 4) ส่งกลับให้หน้าเว็บ — ไม่มี score ติดไปด้วย บอกแค่ระดับ
    return PredictResponse(
        label=info["th"],
        level_key=level_key,
        color=info["color"],
        description=info["description"],
    )
