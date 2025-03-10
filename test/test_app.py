import os
import json
import tempfile
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import app
import crud
import algo  # We patch functions in algo as needed

client = TestClient(app.app)

# --- Override the DB dependency ---
class FakeDB:
    def __init__(self):
        self.users = {}
        self.transactions = []
    def close(self):
        pass

def fake_get_db():
    db = FakeDB()
    try:
        yield db
    finally:
        db.close()

# Override get_db dependency for endpoints that need a DB.
app.app.dependency_overrides[app.get_db] = fake_get_db

# --- Test for root endpoint ---
def test_root():
    response = client.get("/")
    # If no root route is defined, we expect a 404.
    assert response.status_code == 404

# --- Test for /users/ endpoint ---
def fake_create_user(db: Session, username: str):
    return {"id": 1, "username": username}

def test_create_user(monkeypatch):
    monkeypatch.setattr(crud, "create_user", fake_create_user)
    payload = {"username": "testuser"}
    response = client.post("/users/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"
    assert data["id"] == 1

def test_create_user_invalid_payload():
    # Missing required field "username"
    response = client.post("/users/", json={})
    assert response.status_code == 422

# --- Test for /load-transactions/ endpoint ---
def fake_save_transactions_from_json(db: Session, file_path: str = None):
    return {"status": "success", "message": "Transactions loaded successfully."}

def test_load_transactions(monkeypatch):
    monkeypatch.setattr(crud, "save_transactions_from_json", fake_save_transactions_from_json)
    response = client.post("/load-transactions/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Transactions loaded" in data["message"]

def fake_save_transactions_error(db: Session, file_path: str = None):
    return {"status": "error", "message": "Load failed."}

def test_load_transactions_error(monkeypatch):
    monkeypatch.setattr(crud, "save_transactions_from_json", fake_save_transactions_error)
    response = client.post("/load-transactions/")
    # Expect the endpoint to return a 500 error when the status is "error".
    assert response.status_code == 500
    data = response.json()
    assert "Load failed." in data["detail"]

# --- Test for /set-credentials endpoint ---
def test_set_credentials(monkeypatch, tmp_path):
    # Use a temporary file to simulate the .env file.
    temp_env = tmp_path / ".env"
    def fake_open(file, mode):
        return temp_env.open(mode)
    monkeypatch.setattr("builtins.open", fake_open)

    payload = {"username": "testuser", "password": "testpass"}
    response = client.post("/set-credentials", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # Verify that the temporary .env file contains the expected credentials.
    content = temp_env.read_text()
    assert "BANK_USERNAME=testuser" in content
    assert "BANK_PASSWORD=testpass" in content

def test_set_credentials_invalid_payload():
    # Missing "password" field.
    response = client.post("/set-credentials", json={"username": "testuser"})
    assert response.status_code == 422

# --- Test for /analyze-transactions/{user_id} endpoint ---
# Define a fake implementation for analyze_transactions_for_user.
def fake_analyze_transactions(db: Session, user_id: int, eps: float = 0.5, min_samples: int = 5):
    return {"user_id": user_id, "analysis": "dummy analysis"}

def test_analyze_transactions(monkeypatch):
    # Patch the function in the algo module (the endpoint calls this function).
    if not hasattr(algo, "analyze_transactions_for_user"):
        setattr(algo, "analyze_transactions_for_user", fake_analyze_transactions)
    else:
        monkeypatch.setattr(algo, "analyze_transactions_for_user", fake_analyze_transactions)
    response = client.get("/analyze-transactions/1")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["analysis"] == "dummy analysis"

def test_analyze_transactions_error(monkeypatch):
    # Define a fake error function.
    def fake_analyze_error(db: Session, user_id: int, eps: float = 0.5, min_samples: int = 5):
        raise Exception("Test error")
    if not hasattr(algo, "analyze_transactions_for_user"):
        setattr(algo, "analyze_transactions_for_user", fake_analyze_error)
    else:
        monkeypatch.setattr(algo, "analyze_transactions_for_user", fake_analyze_error)
    response = client.get("/analyze-transactions/1")
    assert response.status_code == 500
    data = response.json()
    assert "Test error" in data["detail"]

def test_analyze_transactions_invalid_user_id():
    # Provide a non-integer user_id to trigger a validation error.
    response = client.get("/analyze-transactions/abc")
    assert response.status_code == 422