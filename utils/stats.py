import logging
from datetime import timedelta, datetime, timezone
import os
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, create_engine, func, cast, Date
from sqlalchemy.orm import declarative_base, relationship, Session, sessionmaker

logger = logging.getLogger('app.'+__name__)

engine = create_engine(os.environ.get("DATABASE_URL"), connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False)
    last_login = Column(DateTime, nullable=True)
    last_ip_address = Column(String, nullable=True)
    softban_until = Column(DateTime, nullable=True, default=None)

    usage = relationship("Usage", back_populates="user")

class Usage(Base):
    __tablename__ = "usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, ForeignKey("users.username"), nullable=False)
    time = Column(DateTime, default=datetime.now(timezone.utc))
    token_in = Column(Integer, nullable=False)
    token_out = Column(Integer, nullable=False)
    ip_address = Column(String, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    session_id = Column(String, nullable=True)

    user = relationship("User", back_populates="usage")

def get_usage_statistics():
    db: Session = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        last_24h = now - timedelta(hours=24)

        # Total distinct users in usage table
        total_users = db.query(func.count(func.distinct(Usage.username))).scalar()

        # Total distinct ips in usage table
        total_ips = db.query(func.count(func.distinct(Usage.ip_address))).scalar()

        # Distinct users in last 24 hours
        users_last_24h = db.query(
            func.count(func.distinct(Usage.username))
        ).filter(Usage.time >= last_24h).scalar()

        # Daily token consumption (input and output)
        daily_token_series = db.query(
            func.date(Usage.time).label("day"),
            func.sum(Usage.token_in).label("token_in"),
            func.sum(Usage.token_out).label("token_out")
        ).group_by("day").order_by("day").all()

        # Convert to list of dicts for easy JSON use
        token_series = []
        for row in daily_token_series:
            t_in = row.token_in or 0
            t_out = row.token_out or 0
            t_tot = t_in + t_out
            token_series.append({
                "date": str(row.day),
                "token_in": t_in,
                "token_out": t_out,
                "token_tot": t_tot,
            })

        return {
            "total_users": total_users,
            "total_ips": total_ips,
            "users_last_24h": users_last_24h,
            "daily_token_series": token_series
        }
    except Exception as ex:
        logger.error(f"Failed to calculate statistics: {ex}")
        return None
    finally:
        db.close()
