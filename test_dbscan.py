# test_connection.py
from database import SessionLocal  # import SessionLocal from your database.py
from models import Transaction       # ensure you have defined this in your models.py
from sqlalchemy.orm import Session

def test_connection():
    # Create a new session using SessionLocal
    db: Session = SessionLocal()
    try:
        print("Connected to the database successfully!")
        # Example: Query the count of transactions
        transaction_count = db.query(Transaction).count()
        print(f"Number of transactions in the database: {transaction_count}")
    except Exception as e:
        print(f"Error connecting or querying the database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    test_connection()