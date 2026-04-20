"""
Database layer — SQLAlchemy 2.0 compatible
Stores training results and inference logs in SQLite.
"""
import logging
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Logging
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers= [logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger("AI-Engineering-Suite")

# ORM
Base = declarative_base()

def _now():
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)

class TrainingResult(Base):
    __tablename__ = "training_results"
    id         = Column(Integer, primary_key=True, index=True)
    timestamp  = Column(DateTime(timezone=True), default=_now)
    lambda_val = Column(Float)
    accuracy   = Column(Float)
    sparsity   = Column(Float)
    model_path = Column(String)

class InferenceLog(Base):
    __tablename__ = "inference_logs"
    id          = Column(Integer, primary_key=True, index=True)
    timestamp   = Column(DateTime(timezone=True), default=_now)
    model_id    = Column(String, index=True)
    prediction  = Column(Integer)
    latency_ms  = Column(Float)

# Engine
engine       = create_engine("sqlite:///./metadata.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized.")

def log_training(lam: float, acc: float, sp: float, path: str):
    db  = SessionLocal()
    try:
        db.add(TrainingResult(lambda_val=lam, accuracy=acc, sparsity=sp, model_path=str(path)))
        db.commit()
        logger.info(f"Logged training: λ={lam}, acc={acc:.2f}%, sparsity={sp:.2f}%")
    finally:
        db.close()
