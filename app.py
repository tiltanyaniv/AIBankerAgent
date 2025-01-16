from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import json
from typing import List
import os

app = FastAPI()

DATABASE: str = "bank.db"

def get_db_connection():
    """Get a connection to the current SQLite database."""
    return sqlite3.connect(DATABASE)

def init_db():
    """Initialize the SQLite database and create the table if it doesn't exist."""
    conn: sqlite3.Connection = get_db_connection()
    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bank (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mytransaction INTEGER
        )
    """)
    conn.commit()
    conn.close()

# Initialize the database
init_db()

class Credentials(BaseModel):
    username: str
    password: str

@app.post("/set-credentials")
def set_credentials(credentials: Credentials):
    """
    Save BANK_USERNAME and BANK_PASSWORD to the .env file.
    """
    try:
        # Define the .env file path
        env_file_path = ".env"

        # Write the credentials to the .env file
        with open(env_file_path, "w") as env_file:
            env_file.write(f"BANK_USERNAME={credentials.username}\n")
            env_file.write(f"BANK_PASSWORD={credentials.password}\n")

        # Set the environment variables for the current session (optional)
        os.environ["BANK_USERNAME"] = credentials.username
        os.environ["BANK_PASSWORD"] = credentials.password

        return {"status": "success", "message": "Credentials saved successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/add-transactions")
def add_transactions_from_file():
    """
    Reads the `transactions.json` file, extracts chargedAmount,
    and adds it to the database.
    """
    file_path = "transactions.json"  # File path to the transactions.txt file
    try:
        # Read the file content
        with open(file_path, "r") as file:
            data = json.load(file)  # Parse JSON content
        
        # Open a database connection
        conn: sqlite3.Connection = get_db_connection()
        cursor: sqlite3.Cursor = conn.cursor()
        
        # Iterate through the transactions and add them to the database
        for account in data:
            transactions: List[dict] = account.get("transactions", [])
            for transaction in transactions:
                charged_amount = transaction.get("chargedAmount")
                if charged_amount is not None:
                    cursor.execute("INSERT INTO bank (mytransaction) VALUES (?)", (charged_amount,))
        
        # Commit and close the connection
        conn.commit()
        conn.close()

        return {"status": "success", "message": "Transactions added to the database successfully."}

    except FileNotFoundError:
        return {"status": "error", "message": f"{file_path} not found."}
    except json.JSONDecodeError:
        return {"status": "error", "message": "Error decoding JSON from the file."}
    except Exception as e:
        return {"status": "error", "message": str(e)}