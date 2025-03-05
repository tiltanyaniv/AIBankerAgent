import os
import json
import tempfile
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import app
import crud

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

# Override get_db dependency for testing endpoints that need a DB.
app.app.dependency_overrides[app.get_db] = fake_get_db

# --- Test for root endpoint ---
def test_root():
    response = client.get("/")
    # If your app does not define a root route, expect 404.
    # If you add a root route in the future, update this assertion accordingly.
    assert response.status_code == 404

# --- Test for /users/ endpoint ---
def fake_create_user(db: Session, username: str):
    # Return a fake user dict (or object with attributes)
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
    # Expect FastAPI to return a 422 Unprocessable Entity error.
    assert response.status_code == 422

# --- Test for /load-transactions/ endpoint ---
def fake_save_transactions_from_json(db: Session, file_path: str = None):
    # For testing, simulate a successful load.
    return {"status": "success", "message": "Transactions loaded successfully."}

def test_load_transactions(monkeypatch):
    monkeypatch.setattr(crud, "save_transactions_from_json", fake_save_transactions_from_json)
    response = client.post("/load-transactions/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Transactions loaded" in data["message"]

def fake_save_transactions_error(db: Session, file_path: str = None):
    # Simulate an error condition.
    return {"status": "error", "message": "Load failed."}

def test_load_transactions_error(monkeypatch):
    monkeypatch.setattr(crud, "save_transactions_from_json", fake_save_transactions_error)
    response = client.post("/load-transactions/")
    # Expect the endpoint to return a 500 error when status is "error".
    assert response.status_code == 500
    data = response.json()
    assert "Load failed." in data["detail"]

# --- Test for /set-credentials endpoint ---
def test_set_credentials(monkeypatch, tmp_path):
    # Use a temporary file to simulate .env
    temp_env = tmp_path / ".env"
    # Monkeypatch open so that writes go to our temp file.
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
    # Missing "password"
    response = client.post("/set-credentials", json={"username": "testuser"})
    # Expect a 422 validation error from FastAPI.
    assert response.status_code == 422

# --- Test for /analyze-transactions/{user_id} endpoint ---
def fake_analyze_transactions(db: Session, user_id: int):
    return {"user_id": user_id, "analysis": "dummy analysis"}

def test_analyze_transactions(monkeypatch):
    # Inject fake_analyze_transactions into the crud module's namespace.
    monkeypatch.setitem(crud.__dict__, "analyze_transactions", fake_analyze_transactions)
    response = client.get("/analyze-transactions/1")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["analysis"] == "dummy analysis"

def test_analyze_transactions_error(monkeypatch):
    # Simulate an exception in analyze_transactions.
    def fake_analyze_error(db: Session, user_id: int):
        raise Exception("Test error")
    monkeypatch.setitem(crud.__dict__, "analyze_transactions", fake_analyze_error)
    response = client.get("/analyze-transactions/1")
    assert response.status_code == 500
    data = response.json()
    assert "Test error" in data["detail"]

def test_analyze_transactions_invalid_user_id():
    # Provide a non-integer user_id to trigger a validation error.
    response = client.get("/analyze-transactions/abc")
    assert response.status_code == 422