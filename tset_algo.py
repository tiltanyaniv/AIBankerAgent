import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base  # Use the Base from your database.py
from models import Transaction  # Ensure your Transaction model is defined properly

# Define parsing functions (as before)
def parse_embedding(embedding_data) -> np.ndarray:
    try:
        arr = np.frombuffer(embedding_data, dtype=np.float32)
        return arr
    except Exception as e:
        print(f"Error parsing embedding: {e}")
        return np.array([])

def parse_transactions_df(df: pd.DataFrame) -> pd.DataFrame:
    df["parsed_embedding"] = df["vector_embedding"].apply(parse_embedding)
    df["location_lat"] = df["location_lat"].astype(float)
    df["location_lon"] = df["location_lon"].astype(float)
    df["charged_amount"] = df["charged_amount"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    return df

def get_transactions_for_clustering(db) -> pd.DataFrame:
    query = db.query(
        Transaction.user_id,
        Transaction.vector_embedding,
        Transaction.location_lat,
        Transaction.location_lon,
        Transaction.charged_amount,
        Transaction.date
    )
    df = pd.read_sql(query.statement, db.bind)
    return df

# Pytest fixture: create an in-memory SQLite database and insert dummy data.
@pytest.fixture(scope="module")
def test_db():
    # Use an in-memory SQLite database for testing.
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)  # Create all tables based on your models

    db = TestingSessionLocal()
    
    # Insert dummy transactions into the 'transactions' table.
    dummy_transactions = [
        Transaction(
            user_id=1,
            vector_embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes(),
            location_lat=32.0853,
            location_lon=34.7818,
            charged_amount=100.5,
            date=datetime(2023, 1, 2, 10, 0, 0)
        ),
        Transaction(
            user_id=2,
            vector_embedding=np.array([1.1, 1.2, 1.3], dtype=np.float32).tobytes(),
            location_lat=31.7683,
            location_lon=35.2137,
            charged_amount=200.7,
            date=datetime(2023, 5, 6, 15, 30, 0)
        )
    ]
    db.add_all(dummy_transactions)
    db.commit()
    
    yield db  # Provide the test database session.
    
    db.close()

# Test function using pytest.
def test_get_and_parse_transactions(test_db):
    # Step 1: Query the transactions.
    df = get_transactions_for_clustering(test_db)
    assert len(df) == 2, "Should retrieve 2 transactions from the dummy data."
    
    # Step 2: Parse the DataFrame.
    df_parsed = parse_transactions_df(df)
    
    # Check that new columns are added.
    for col in ["parsed_embedding", "year", "month", "day"]:
        assert col in df_parsed.columns, f"Column '{col}' not found in parsed DataFrame."
    
    # Validate the parsed embedding shape for the first row.
    first_emb = df_parsed.iloc[0]["parsed_embedding"]
    assert isinstance(first_emb, np.ndarray), "Parsed embedding is not a NumPy array."
    assert first_emb.shape == (3,), "Parsed embedding should have shape (3,)"
    
    # Validate date extraction for the first row.
    row0 = df_parsed.iloc[0]
    assert row0["year"] == 2023, "Year not parsed correctly for row 0."
    assert row0["month"] == 1, "Month not parsed correctly for row 0."
    assert row0["day"] == 2, "Day not parsed correctly for row 0."
    
    # Optionally, print out the parsed DataFrame for manual inspection.
    print("\nParsed DataFrame:")
    print(df_parsed.head())