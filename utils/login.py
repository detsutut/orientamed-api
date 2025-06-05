import logging
import jwt
from datetime import timedelta, datetime, timezone
import os
import yaml
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, select, create_engine, func, cast, Date
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import declarative_base, relationship, Session, sessionmaker

logger = logging.getLogger('app.'+__name__)

engine = create_engine(os.environ.get("DATABASE_URL"), connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# SQLAlchemy model to map to existing table
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

with open(os.getenv("API_SETTINGS_PATH")) as stream:
    api_config = yaml.safe_load(stream)

def authenticate(username: str, password: str):
    logger.debug(f"Authenticating user {username}...")
    db: Session = SessionLocal()
    try:
        stmt = select(User).where(User.username == username)
        user = db.scalars(stmt).first()
        if user and password==user.password:
            user.last_login = datetime.now(timezone.utc)
            db.commit()
            return True
        else:
            logger.warning(f"User {username} not found for last_login update")
            return False
    except Exception as e:
        db.rollback()
        logger.error(f"Authentication error for {username}: {e}")
        return False
    finally:
        db.close()

def create_access_token(username: str, expires_delta: timedelta | None = None):
    logger.info(f"Creating new access token for user {username}...")
    to_encode = {"usr": username}
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=api_config.get("access-token-expire-minutes",15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.environ.get("SECRET_KEY"), algorithm=api_config.get("algorithm",'HS256'))
    return encoded_jwt

def update_user(username: str, last_ip_address: str | None, softban_until: datetime | None):
    db: Session = SessionLocal()
    try:
        stmt = select(User).where(User.username == username)
        user = db.scalars(stmt).first()
        if user:
            if softban_until:
                user.softban_until = softban_until
            if last_ip_address:
                user.last_ip_address = last_ip_address
            db.commit()
            return True
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user info for {username}: {e}")
        return False
    finally:
        db.close()

def log_usage(username: str, token_in: int, token_out: int, **kwargs):
    db: Session = SessionLocal()
    try:
        valid_fields = {c.key for c in inspect(Usage).attrs}
        safe_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        entry = Usage(
            username=username,
            token_in=token_in,
            token_out=token_out,
            **safe_kwargs
        )
        db.add(entry)
        db.commit()
        logger.debug(f"Logged usage for {username}")
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to log usage: {e}")
        return False
    finally:
        db.close()

def verify_token(token: str):
    logger.debug(f"Veryfing token {token}...")
    try:
        payload = jwt.decode(token, os.environ.get("SECRET_KEY"), algorithms=[api_config.get("algorithm",'HS256')])
        return payload.get("usr")
    except jwt.ExpiredSignatureError as e:
        logger.error(e)
        return None
    except jwt.DecodeError as e:
        logger.error(e)
        return None
    except Exception as e:
        logger.error(e)
        return None

def check_ban(username: str):
    logger.debug(f"Checking {username}'s ban...")
    db: Session = SessionLocal()
    try:
        stmt = select(User).where(User.username == username)
        user = db.scalars(stmt).first()
        if user.softban_until and user.softban_until > datetime.now(timezone.utc):
            logger.warning(f"User {username} is banned until {user.softban_until}")
            return True
        else:
            logger.debug(f"User {username} is not banned.")
            return False
    except Exception as e:
        logger.error(f"Error checking ban for {username}: {e}")
        return True
    finally:
        db.close()

def check_daily_token_limit(username: str, role="preview"):
    if role == "preview":
        limit = 10000
    elif role == "admin":
        logger.warning(f"Checking limit for admin role: this is not supposed to happen.")
        limit = 500000
    elif role == "user":
        limit = 20000
    else:
        logger.error(f"Invalid role: {role}")
        return True
    db: Session = SessionLocal()
    today = datetime.now(timezone.utc).date()
    try:
        total_tokens = db.query(
            func.coalesce(func.sum(Usage.token_in + Usage.token_out), 0)
        ).filter(
            Usage.username == username,
            cast(Usage.time, Date) == today
        ).scalar()
        return total_tokens >= limit
    except Exception as e:
        logger.error(f"Error checking daily limit for {username}: {e}")
        return True
    finally:
        db.close()

def set_softban(username: str, hours: int = 24) -> bool:
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user:
            user.softban_until = datetime.now(timezone.utc) + timedelta(hours=hours)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to set softban for user {username}: {e}")
        return False
    finally:
        db.close()

def login(username: str, password: str):
    if authenticate(username, password):
        token = create_access_token(username)
        logger.debug(f"Successfully authenticated {username}...")
    else:
        logger.debug(f"Authentication failed")
        token = None
    return token

def get_role(username: str):
    logger.debug(f"Checking the role of {username}...")
    db: Session = SessionLocal()
    try:
        stmt = select(User).where(User.username == username)
        user = db.scalars(stmt).first()
        if user:
            return user.role
        else:
            logger.warning(f"User {username} not found for last_login update")
            return None
    except Exception as e:
        db.rollback()
        logger.error(f"Error checking role {username} role: {e}")
        return None
    finally:
        db.close()
