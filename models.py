from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime, LargeBinary
from sqlalchemy.orm import relationship
from database import Base
import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    transactions = relationship("Transaction", back_populates="user")

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    charged_amount = Column(Float)
    description = Column(String)
    category = Column(String)
    date = Column(DateTime, default=datetime.datetime.utcnow)
    location = Column(String, nullable=True)  
    location_lat = Column(Float, nullable=True)  
    location_lon = Column(Float, nullable=True)  
    vector_embedding = Column(LargeBinary, nullable=True)
    original_currency = Column(String)

    user = relationship("User", back_populates="transactions")