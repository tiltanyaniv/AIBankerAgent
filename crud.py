import json
from sqlalchemy.orm import Session
from models import User, Transaction
from datetime import datetime

def create_user(db: Session, account_number: str):
    """Create a new user if they don't exist."""
    existing_user = db.query(User).filter(User.username == account_number).first()
    if existing_user:
        return existing_user  # ✅ Prevent duplicate users

    user = User(username=account_number)  # ✅ Store accountNumber as username
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def save_transactions_from_json(db: Session, file_path: str = "transactions.json"):
    """Reads transactions from JSON and saves them to the database."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for account in data:
            account_number = str(account.get("accountNumber")).strip()  # ✅ Ensure consistency

            # ✅ FIX: Query using account_number and prevent duplicate users
            user = db.query(User).filter(User.username == account_number).first()

            if not user:
                user = User(username=account_number)
                db.add(user)
                db.commit()  # ✅ Commit so the user is available immediately
                db.refresh(user)

            for transaction in account.get("transactions", []):
                transaction_date = datetime.strptime(transaction["date"], "%Y-%m-%dT%H:%M:%S.%fZ")

                new_transaction = Transaction(
                    user_id=user.id,  # ✅ Always use the existing user ID
                    charged_amount=transaction["chargedAmount"],
                    description=transaction["description"],
                    category=transaction["category"],
                    date=transaction_date,
                    original_currency=transaction["originalCurrency"]
                )

                db.add(new_transaction)

        db.commit()
        return {"status": "success", "message": "Transactions saved to the database."}

    except Exception as e:
        db.rollback()  # ✅ Prevents partial commits in case of failure
        return {"status": "error", "message": str(e)}