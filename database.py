import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# อ่าน connection string จาก environment variable
# ตัวอย่างรูปแบบ: postgresql://user:password@host:5432/dbname
# บน Render: ไปที่ Dashboard > PostgreSQL instance > คัดลอก "Internal Database URL" หรือ "External Database URL"
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/stress_scouter"
)

# Render ให้ URL แบบ "postgres://" (เก่า) ต้องแปลงเป็น "postgresql://" ให้ SQLAlchemy อ่านได้
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency สำหรับ FastAPI: เปิด session แล้วปิดอัตโนมัติหลังใช้เสร็จ"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
