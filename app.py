from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import crud
from pydantic import BaseModel
import datetime
import os

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models
class UserCreate(BaseModel):
    username: str

class TransactionCreate(BaseModel):
    user_id: int
    charged_amount: float
    description: str
    category: str
    date: datetime.datetime
    original_currency: str

@app.post("/users/")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db, user.username)


class Credentials(BaseModel):
    username: str
    password: str

@app.post("/load-transactions/")
def load_transactions(db: Session = Depends(get_db)):
    """Reads transactions from JSON and saves them to the database."""
    result = crud.save_transactions_from_json(db)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/set-credentials")
def set_credentials(credentials: Credentials):
    """
    Save BANK_USERNAME and BANK_PASSWORD to the .env file.
    """
    try:
        env_file_path = ".env"
        with open(env_file_path, "w") as env_file:
            env_file.write(f"BANK_USERNAME={credentials.username}\n")
            env_file.write(f"BANK_PASSWORD={credentials.password}\n")

        os.environ["BANK_USERNAME"] = credentials.username
        os.environ["BANK_PASSWORD"] = credentials.password

        return {"status": "success", "message": "Credentials saved successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/detect-unusual-transactions/")
def detect_unusual(db: Session = Depends(get_db), threshold: float = 0.2):
    """
    Detect unusual transactions based on transaction vectors using OpenAI embeddings.
    """
    return crud.detect_unusual_transactions(db, threshold)