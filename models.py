from sqlalchemy import Column, Integer, String, TIMESTAMP, func
from database import Base


class StressResult(Base):
    """
    ผลการประเมินแต่ละครั้ง ไม่เก็บชื่อ/รหัสนิสิต/คณะ
    respondent_code คือเลขอ้างอิงเรียงลำดับอัตโนมัติ เช่น 00001, 00002, ...
    (สร้างจาก id หลัง insert แล้วเติม 0 ข้างหน้าให้ครบ 5 หลัก)
    """
    __tablename__ = "stress_results"

    id = Column(Integer, primary_key=True, index=True)
    respondent_code = Column(String(10), unique=True, index=True, nullable=False)
    score = Column(Integer, nullable=False)          # เก็บคะแนน 0-100 แบบละเอียดในฐานข้อมูล
    level = Column(String(20), nullable=False)        # low / medium / high / highest
    submitted_at = Column(TIMESTAMP, server_default=func.now())
