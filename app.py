from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import json
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
    # Create table with columns for each month
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monthly_expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            jan REAL DEFAULT NULL,
            feb REAL DEFAULT NULL,
            mar REAL DEFAULT NULL,
            apr REAL DEFAULT NULL,
            may REAL DEFAULT NULL,
            jun REAL DEFAULT NULL,
            jul REAL DEFAULT NULL,
            aug REAL DEFAULT NULL,
            sep REAL DEFAULT NULL,
            oct REAL DEFAULT NULL,
            nov REAL DEFAULT NULL,
            dec REAL DEFAULT NULL
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
    Reads the `transactions.json` file, extracts chargedAmount and date,
    and adds expenses to the appropriate month columns in the database.
    """
    file_path = "transactions.json"  # File path to the transactions.json file
    try:
        # Read the file content
        with open(file_path, "r") as file:
            data = json.load(file)  # Parse JSON content
        
        # Open a database connection
        conn: sqlite3.Connection = get_db_connection()
        cursor: sqlite3.Cursor = conn.cursor()
        
        # Iterate through the transactions
        for account in data:  # Assuming data is a list of accounts
            transactions = account.get("transactions", [])
            for transaction in transactions:
                charged_amount = transaction.get("chargedAmount")
                transaction_date = transaction.get("date")  # Get the transaction date

                if charged_amount is not None and transaction_date:
                    # Parse the month from the date
                    month = int(transaction_date.split("-")[1])  # Extract the month as an integer
                    
                    # Map the month to the correct column name
                    month_column = ["jan", "feb", "mar", "apr", "may", "jun", 
                                    "jul", "aug", "sep", "oct", "nov", "dec"][month - 1]

                    # Insert the expense into the appropriate column
                    cursor.execute(f"""
                        INSERT INTO monthly_expenses ({month_column})
                        VALUES (?)
                    """, (charged_amount,))
        
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


class MonthRequest(BaseModel):
    month: str

@app.post("/get-monthly-summary")
def get_monthly_summary(request: MonthRequest):
    """
    Get the total income (positive expenses) and outcome (negative expenses) for a given month.
    The input month is case-insensitive (e.g., "jan", "JAN", "Jan").
    """
    # Validate the input month
    valid_months = ["jan", "feb", "mar", "apr", "may", "jun", 
                    "jul", "aug", "sep", "oct", "nov", "dec"]
    month = request.month.lower()  # Convert the input month to lowercase

    if month not in valid_months:
        raise HTTPException(status_code=400, detail="Invalid month. Use: jan, feb, mar, etc.")

    try:
        # Open a database connection
        conn: sqlite3.Connection = get_db_connection()
        cursor: sqlite3.Cursor = conn.cursor()
        
        # Query to calculate income (positive) and outcome (negative) for the given month
        cursor.execute(f"""
            SELECT
                SUM(CASE WHEN {month} > 0 THEN {month} ELSE 0 END) AS income,
                SUM(CASE WHEN {month} < 0 THEN {month} ELSE 0 END) AS outcome
            FROM monthly_expenses
        """)
        result = cursor.fetchone()

        # Close the database connection
        conn.close()

        # Return the results as a dictionary
        income = result[0] or 0  # Default to 0 if no income
        outcome = result[1] or 0  # Default to 0 if no outcome
        return {"month": month, "income": income, "outcome": outcome}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")