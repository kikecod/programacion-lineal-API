from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
from passlib.context import CryptContext
import os

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Hi-alsoWm24@localhost/linear_programming_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password Hashing Configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Database Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)  # Cambiado de 'password' a 'password_hash'
    created_at = Column(DateTime, default=datetime.utcnow)
    problems = relationship("ProblemSolution", back_populates="user")

class ProblemSolution(Base):
    __tablename__ = "problem_solutions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    problem_type = Column(String)
    function_objective = Column(String)  # Almacena como JSON string
    restrictions = Column(String)  # Almacena como JSON string
    solution_value = Column(Float)
    solution_variables = Column(String)  # Almacena como JSON string
    constraints = Column(String)  # Almacena como JSON string
    coefficient_ranges = Column(String)  # Almacena como JSON string
    rhs_ranges = Column(String)  # Almacena como JSON string
    solved_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="problems")

# Create tables function
def create_tables():
    Base.metadata.create_all(bind=engine)

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()