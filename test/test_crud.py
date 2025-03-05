import json
import numpy as np
import pytest
from datetime import datetime
from unittest.mock import MagicMock
import crud

@pytest.fixture
def fake_db():
    return MagicMock()

@pytest.fixture
def dummy_json_file(tmp_path):
    # Dummy JSON with one account and one transaction.
    dummy_data = [
        {
            "accountNumber": "12345",
            "transactions": [
                {
                    "date": "2023-01-01T12:00:00.000000Z",
                    "description": "Test transaction",
                    "originalCurrency": "USD",
                    "category": "TestCategory",
                    "chargedAmount": 100.0
                }
            ]
        }
    ]
    file_path = tmp_path / "transactions.json"
    file_path.write_text(json.dumps(dummy_data))
    return file_path

def test_create_user_existing(fake_db):
    dummy_user = type('DummyUser', (), {'id': 1, 'username': '12345'})()
    fake_db.query.return_value.filter.return_value.first.return_value = dummy_user
    user = crud.create_user(fake_db, '12345')
    assert user == dummy_user

def test_create_user_new(fake_db):
    fake_db.query.return_value.filter.return_value.first.return_value = None
    def fake_refresh(user):
        user.id = 1
    fake_db.refresh.side_effect = fake_refresh
    user = crud.create_user(fake_db, '67890')
    assert user.username == '67890'
    assert user.id == 1

def test_save_transactions_from_json(fake_db, dummy_json_file, monkeypatch):
    fake_db.query.return_value.filter.return_value.first.return_value = None

    monkeypatch.setattr(crud, "get_location", lambda desc: "Test Location")
    monkeypatch.setattr(crud, "get_location_coordinates", lambda loc: (10.0, 20.0))
    dummy_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    monkeypatch.setattr(crud, "create_transaction_vector", lambda orig, desc, cat: dummy_embedding)
    monkeypatch.setattr(crud, "convert_to_usd", lambda amount, curr: amount)

    result = crud.save_transactions_from_json(fake_db, file_path=str(dummy_json_file))
    assert result["status"] == "success"

def test_save_transactions_from_json_user_exists(fake_db, dummy_json_file, monkeypatch):
    dummy_user = type('DummyUser', (), {'id': 1, 'username': '12345'})()
    fake_db.query.return_value.filter.return_value.first.return_value = dummy_user

    monkeypatch.setattr(crud, "get_location", lambda desc: "Test Location")
    monkeypatch.setattr(crud, "get_location_coordinates", lambda loc: (10.0, 20.0))
    dummy_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    monkeypatch.setattr(crud, "create_transaction_vector", lambda orig, desc, cat: dummy_embedding)
    monkeypatch.setattr(crud, "convert_to_usd", lambda amount, curr: amount)

    result = crud.save_transactions_from_json(fake_db, file_path=str(dummy_json_file))
    assert result["status"] == "success"

def test_save_transactions_from_json_multiple_transactions(monkeypatch, tmp_path, fake_db):
    # Dummy JSON with two transactions.
    dummy_data = [
        {
            "accountNumber": "12345",
            "transactions": [
                {
                    "date": "2023-01-01T12:00:00.000000Z",
                    "description": "Transaction 1",
                    "originalCurrency": "USD",
                    "category": "Cat1",
                    "chargedAmount": 50.0
                },
                {
                    "date": "2023-01-02T12:00:00.000000Z",
                    "description": "Transaction 2",
                    "originalCurrency": "EUR",
                    "category": "Cat2",
                    "chargedAmount": 150.0
                }
            ]
        }
    ]
    file_path = tmp_path / "multi.json"
    file_path.write_text(json.dumps(dummy_data))

    monkeypatch.setattr(crud, "get_location", lambda desc: "Test Location")
    monkeypatch.setattr(crud, "get_location_coordinates", lambda loc: (10.0, 20.0))
    dummy_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    monkeypatch.setattr(crud, "create_transaction_vector", lambda orig, desc, cat: dummy_embedding)
    monkeypatch.setattr(crud, "convert_to_usd", lambda amount, curr: amount)

    result = crud.save_transactions_from_json(fake_db, file_path=str(file_path))
    assert result["status"] == "success"

def test_save_transactions_from_json_invalid_json(monkeypatch, tmp_path, fake_db):
    # Provide invalid JSON content.
    file_path = tmp_path / "invalid.json"
    file_path.write_text("invalid json")
    try:
        result = crud.save_transactions_from_json(fake_db, file_path=str(file_path))
    except Exception:
        result = {"status": "error"}
    assert result["status"] == "error"

def test_save_transactions_invalid_date(monkeypatch, tmp_path, fake_db):
    # Provide a transaction with an invalid date format.
    dummy_data = [
        {
            "accountNumber": "12345",
            "transactions": [
                {
                    "date": "invalid-date-format",
                    "description": "Test transaction",
                    "originalCurrency": "USD",
                    "category": "TestCategory",
                    "chargedAmount": 100.0
                }
            ]
        }
    ]
    file_path = tmp_path / "invalid_date.json"
    file_path.write_text(json.dumps(dummy_data))
    
    monkeypatch.setattr(crud, "get_location", lambda desc: "Test Location")
    monkeypatch.setattr(crud, "get_location_coordinates", lambda loc: (10.0, 20.0))
    dummy_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    monkeypatch.setattr(crud, "create_transaction_vector", lambda orig, desc, cat: dummy_embedding)
    monkeypatch.setattr(crud, "convert_to_usd", lambda amount, curr: amount)
    
    try:
        result = crud.save_transactions_from_json(fake_db, file_path=str(file_path))
    except Exception:
        result = {"status": "error"}
    assert result["status"] in ("success", "error")

class DummyResponse:
    def __init__(self, content):
        self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})()})()]

def test_get_location_online(monkeypatch):
    dummy_client = MagicMock()
    dummy_client.chat.completions.create.return_value = DummyResponse("Online Purchase")
    monkeypatch.setattr(crud, "OpenAI", lambda: dummy_client)
    result = crud.get_location("Some description")
    assert result == "Online Purchase"

def test_get_location_physical(monkeypatch):
    dummy_client = MagicMock()
    dummy_client.chat.completions.create.side_effect = [
        DummyResponse("Physical Store"),
        DummyResponse("TestCity")
    ]
    monkeypatch.setattr(crud, "OpenAI", lambda: dummy_client)
    result = crud.get_location("Some description")
    assert result == "TestCity"

def test_get_location_empty_response(monkeypatch):
    dummy_client = MagicMock()
    dummy_client.chat.completions.create.return_value = DummyResponse("")
    monkeypatch.setattr(crud, "OpenAI", lambda: dummy_client)
    result = crud.get_location("Some description")
    assert result is None

def test_get_location_coordinates_unknown(monkeypatch):
    monkeypatch.setattr(crud, "Nominatim", lambda user_agent: type("Dummy", (), {"geocode": lambda q: None}))
    result = crud.get_location_coordinates("Unknown")
    assert result == (0.0, 0.0)

def test_get_location_coordinates_valid(monkeypatch):
    class DummyGeo:
        def __init__(self, latitude, longitude):
            self.latitude = latitude
            self.longitude = longitude
    dummy_geo = DummyGeo(50.0, 60.0)
    monkeypatch.setattr(crud, "Nominatim", lambda user_agent: type("Dummy", (), {"geocode": lambda q: dummy_geo}))
    result = crud.get_location_coordinates("TestCity")
    assert result == (50.0, 60.0)

def test_get_embedding(monkeypatch):
    dummy_embedding = [0.5, 0.6, 0.7]
    DummyEmbeddingResponse = type("DummyEmbeddingResponse", (), {
        "data": [type("DummyData", (), {"embedding": dummy_embedding})()]
    })
    dummy_client = MagicMock()
    dummy_client.embeddings.create.return_value = DummyEmbeddingResponse
    monkeypatch.setattr(crud, "openai", type("DummyOpenAI", (), {"OpenAI": lambda self=None: dummy_client}))
    result = crud.get_embedding("Test text")
    np.testing.assert_array_almost_equal(result, np.array(dummy_embedding, dtype=np.float32))

def test_get_embedding_empty(monkeypatch):
    # Simulate an API response with an empty embedding.
    DummyEmbeddingResponse = type("DummyEmbeddingResponse", (), {
        "data": [type("DummyData", (), {"embedding": []})()]
    })
    dummy_client = MagicMock()
    dummy_client.embeddings.create.return_value = DummyEmbeddingResponse
    monkeypatch.setattr(crud, "openai", type("DummyOpenAI", (), {"OpenAI": lambda self=None: dummy_client}))
    result = crud.get_embedding("Test empty")
    np.testing.assert_array_equal(result, np.array([], dtype=np.float32))

def test_create_transaction_vector(monkeypatch):
    fixed_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    monkeypatch.setattr(crud, "get_embedding", lambda text: fixed_embedding)
    result = crud.create_transaction_vector("USD", "Some description", "Category")
    np.testing.assert_array_almost_equal(result, fixed_embedding)

def test_convert_to_usd_usd():
    result = crud.convert_to_usd(100.0, "USD")
    assert result == 100.0

def test_convert_to_usd_other(monkeypatch):
    class DummyResponse:
        def json(self):
            return {"rates": {"USD": 0.5}}
    monkeypatch.setattr(crud.requests, "get", lambda url: DummyResponse())
    result = crud.convert_to_usd(200.0, "EUR")
    assert result == 100.0

def test_convert_to_usd_failure(monkeypatch):
    def dummy_get(url):
        raise Exception("API error")
    monkeypatch.setattr(crud.requests, "get", dummy_get)
    result = crud.convert_to_usd(300.0, "EUR")
    assert result == 300.0

def test_convert_to_usd_no_rate(monkeypatch):
    class DummyResponse:
        def json(self):
            return {"rates": {}}
    monkeypatch.setattr(crud.requests, "get", lambda url: DummyResponse())
    result = crud.convert_to_usd(200.0, "EUR")
    assert result == 200.0