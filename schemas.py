from pydantic import BaseModel


class PredictRequest(BaseModel):
    # 20 features จากแบบสอบถาม (ตรงกับ FIELD_IDS ใน index2.html)
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


class PredictResponse(BaseModel):
    # ไม่ส่งคะแนน (score) กลับไปให้ผู้ตอบแบบสอบถามเห็น บอกแค่ระดับ
    label: str          # ข้อความภาษาไทย เช่น "ความเครียดสูง"
    level_key: str       # low / medium / high / highest
    color: str
    description: str
