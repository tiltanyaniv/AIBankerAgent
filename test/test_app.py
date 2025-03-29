import os
import json
import tempfile
import pytest
import subprocess
import numpy as np
import pandas as pd
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
    # Create a temporary file simulating index.js with proper content.
    temp_index = tmp_path / "index.js"
    initial_content = (
        'let BANK_USERNAME = "olduser";\n'
        'let BANK_PASSWORD = "oldpass";'
    )
    temp_index.write_text(initial_content)

    def fake_open(file, mode):
        if file == "index.js":
            return temp_index.open(mode)
        return open(file, mode)
    monkeypatch.setattr("builtins.open", fake_open)

    payload = {"username": "testuser", "password": "testpass"}
    response = client.post("/set-credentials", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    # Verify that the temporary index.js file has updated credentials.
    content = temp_index.read_text()
    assert 'let BANK_USERNAME = "testuser"' in content
    assert 'let BANK_PASSWORD = "testpass"' in content

def test_set_credentials_invalid_payload():
    # Missing "password" field.
    response = client.post("/set-credentials", json={"username": "testuser"})
    assert response.status_code == 422

# --- Test for /get-credentials endpoint ---
def test_get_credentials(monkeypatch, tmp_path):
    temp_index = tmp_path / "index.js"
    temp_index.write_text('let BANK_USERNAME = "user";\nlet BANK_PASSWORD = "pass";')
    def fake_open(file, mode):
        if file == "index.js":
            return temp_index.open(mode)
        return open(file, mode)
    monkeypatch.setattr("builtins.open", fake_open)
    response = client.get("/get-credentials")
    assert response.status_code == 200
    data = response.json()
    assert data["BANK_USERNAME"] == "user"
    assert data["BANK_PASSWORD"] == "pass"

# --- Test for /get-transData endpoint ---
def test_get_transData(monkeypatch, tmp_path):
    temp_index = tmp_path / "index.js"
    temp_index.write_text("companyId: CompanyTypes.TEST_ID\nstartDate: new Date('2020-01-01')")
    def fake_open(file, mode):
        if file == "index.js":
            return temp_index.open(mode)
        return open(file, mode)
    monkeypatch.setattr("builtins.open", fake_open)
    response = client.get("/get-transData")
    assert response.status_code == 200
    data = response.json()
    assert data["company_id"] == "TEST_ID"
    assert data["start_date"] == "2020-01-01"

# --- Test for /set-transData endpoint ---
def test_set_transData(monkeypatch, tmp_path):
    temp_index = tmp_path / "index.js"
    temp_index.write_text("companyId: CompanyTypes.OLD_ID\nstartDate: new Date('2020-01-01')")
    def fake_open(file, mode):
        if file == "index.js":
            return temp_index.open(mode)
        return open(file, mode)
    monkeypatch.setattr("builtins.open", fake_open)
    payload = {"company_id": "NEW_ID", "start_date": "2021-01-01"}
    response = client.post("/set-transData", json=payload)
    assert response.status_code == 200
    content = temp_index.read_text()
    assert "CompanyTypes.NEW_ID" in content
    assert "new Date('2021-01-01')" in content

# --- Test for /run-transaction endpoint ---
def test_run_transaction_success(monkeypatch):
    def fake_run(cmd, capture_output, text):
        class FakeResult:
            returncode = 0
            stdout = "Success"
            stderr = ""
            args = cmd
        return FakeResult()
    monkeypatch.setattr(subprocess, "run", fake_run)
    response = client.post("/run-transaction")
    assert response.status_code == 200
    data = response.json()
    assert "Success" in data["stdout"]

def test_run_transaction_error(monkeypatch):
    def fake_run(cmd, capture_output, text):
        class FakeResult:
            returncode = 1
            stdout = "Error occurred"
            stderr = "Some error"
            args = cmd
        return FakeResult()
    monkeypatch.setattr(subprocess, "run", fake_run)
    response = client.post("/run-transaction")
    assert response.status_code == 500
    data = response.json()
    assert "Error running index.js" in data["detail"]

# --- Fake functions for analysis endpoints ---
def fake_analyze_transactions(db: Session, user_id: int, grid_search: bool = True, default_eps: float = 0.5, default_min_samples: int = 5):
    return {"user_id": user_id, "analysis": "dummy analysis", "eps_used": default_eps, "min_samples_used": default_min_samples}

def fake_analyze_error(db: Session, user_id: int, grid_search: bool = True, default_eps: float = 0.5, default_min_samples: int = 5):
    raise Exception("Test error")

# --- Test for /analyze-transactions/{user_id} endpoint ---
def test_analyze_transactions(monkeypatch):
    monkeypatch.setattr(algo, "analyze_transactions_for_user", fake_analyze_transactions)
    response = client.get("/analyze-transactions/1")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["analysis"] == "dummy analysis"

def test_analyze_transactions_error(monkeypatch):
    monkeypatch.setattr(algo, "analyze_transactions_for_user", fake_analyze_error)
    response = client.get("/analyze-transactions/1")
    assert response.status_code == 500
    data = response.json()
    # Check that the error message contains "Test error"
    assert "Test error" in data["detail"]

def test_analyze_transactions_invalid_user_id():
    # Provide a non-integer user_id to trigger a validation error.
    response = client.get("/analyze-transactions/abc")
    assert response.status_code == 422

# --- Test for /visualize-transactions/{user_id} endpoint ---
def test_visualize_transactions(monkeypatch):
    # Patch the analysis and clustering functions to return dummy values.
    monkeypatch.setattr(
        algo,
        "analyze_transactions_for_user",
        lambda db, user_id, grid_search=True, default_eps=0.5, default_min_samples=5: {"eps_used": 0.5, "min_samples_used": 5}
    )
    df_dummy = pd.DataFrame({"dummy": [1, 2, 3]})
    monkeypatch.setattr(algo, "get_transactions_for_clustering", lambda db, user_id: df_dummy)
    monkeyatch_parse = monkeypatch.setattr(algo, "parse_transactions_df", lambda df: df)
    dummy_array = np.array([[1, 2], [3, 4], [5, 6]])
    monkeypatch.setattr(algo, "build_feature_matrix", lambda df: (dummy_array, [1, 2, 3]))
    monkeypatch.setattr(algo, "scale_features", lambda X: X)
    monkeypatch.setattr(algo, "run_dbscan", lambda X, eps, min_samples: np.array([0, 1, 0]))
    
    response = client.get("/visualize-transactions/1")
    assert response.status_code == 200
    # Verify that the response returns a PNG image.
    assert response.headers["content-type"] == "image/png"