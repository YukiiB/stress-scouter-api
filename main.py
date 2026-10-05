import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import engine, get_db, Base
import models
from schemas import PredictRequest, PredictResponse, FactorOut
from advice import FACTOR_TIPS

# สร้างตารางในฐานข้อมูลอัตโนมัติ (ถ้ายังไม่มี) ตอนแอปเริ่มทำงาน
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # โปรดักชันจริงควรระบุ origin ของหน้าเว็บให้ชัดเจน
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดไฟล์ที่เทรนไว้ (model.pkl) จาก train_ml_composite.py
artifact = joblib.load("model.pkl")

MODEL = artifact["model"]
FEATURE_ORDER = artifact["feature_order"]
LABELS = artifact["labels"]  # {0:"low",1:"medium",2:"high",3:"highest"}

# พารามิเตอร์คำนวณ ml_score จาก predict_proba (ใช้ตัดสินระดับ)
MU_LOW, SD_LOW = artifact["ml_mu_low"], artifact["ml_sd_low"]
MU_HIGH, SD_HIGH = artifact["ml_mu_high"], artifact["ml_sd_high"]
CUTS = artifact["ml_cuts"]

# พารามิเตอร์ raw-feature z-score (ใช้แค่หา top_factors อธิบายเหตุผล)
POS_COLS = artifact["index_pos_cols"]
NEG_COLS = artifact["index_neg_cols"]
COL_MEANS = artifact["index_col_means"]
COL_STDS = artifact["index_col_stds"]
FEATURE_LABELS_TH = artifact["feature_labels_th"]

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


def compute_ml_score(row: pd.DataFrame) -> float:
    """predict_proba -> z-score composite (p_high - p_low) -> sigmoid 0-100"""
    proba = MODEL.predict_proba(row)[0]          # [p_low, p_medium, p_high]
    z_high = (proba[2] - MU_HIGH) / SD_HIGH
    z_low = (proba[0] - MU_LOW) / SD_LOW
    raw = z_high - z_low
    return float(100 / (1 + np.exp(-raw)))


def compute_top_factors(row: pd.DataFrame, n: int = 3) -> list[FactorOut]:
    """จัดอันดับปัจจัยที่ 'ทำให้เครียดขึ้น' มากที่สุดเทียบกับค่าเฉลี่ยของกลุ่มตัวอย่าง"""
    scores = {}
    for col in POS_COLS + NEG_COLS:
        sd = COL_STDS[col] or 1.0
        z = (float(row[col].iloc[0]) - COL_MEANS[col]) / sd
        scores[col] = z if col in POS_COLS else -z   # NEG: ค่าต่ำ = เครียดขึ้น
    top = sorted(scores, key=scores.get, reverse=True)[:n]
    return [
        FactorOut(label=FEATURE_LABELS_TH.get(c, c), tip=FACTOR_TIPS.get(c, ""))
        for c in top if scores[c] > 0
    ]


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Stress Scouter API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, db: Session = Depends(get_db)):
    # 1) เตรียม features ตามลำดับที่โมเดลต้องการ
    row = pd.DataFrame([{name: getattr(payload, name) for name in FEATURE_ORDER}])

    # 2) ml_score จากผลโมเดล แล้วแบ่ง 4 ระดับด้วย cut-off
    ml_score = compute_ml_score(row)
    level_code = int(np.searchsorted(CUTS, ml_score, side="right"))  # 0-3
    level_key = LABELS[level_code]
    info = LEVEL_INFO[level_key]
    score = int(round(ml_score))  # เก็บลง DB เท่านั้น ไม่ส่งกลับให้ผู้ใช้เห็น

    # 3) ปัจจัยที่เกี่ยวข้องสูงสุด
    factors = compute_top_factors(row)

    # 4) บันทึกผลลง database — ไม่เก็บชื่อ/รหัสนิสิต/คณะ ใช้ respondent_code แทน
    result = models.StressResult(
        respondent_code="",  # อัปเดตด้านล่างหลังรู้ id จริง
        score=score,
        level=level_key,
    )
    db.add(result)
    db.flush()

    result.respondent_code = f"{result.id:05d}"
    db.commit()

    # 5) ส่งกลับให้หน้าเว็บ
    return PredictResponse(
        label=info["th"],
        level_key=level_key,
        color=info["color"],
        description=info["description"],
        top_factors=factors,
    )
