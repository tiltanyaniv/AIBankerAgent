import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock
from sqlalchemy.orm import Session

# Ensure the project root is in sys.path if needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from your module.
from algo import (
    get_transactions_for_clustering,
    parse_embedding,
    parse_transactions_df,
    build_feature_matrix,
    scale_features
)

def test_get_transactions_for_clustering(monkeypatch):
    # Create a dummy DataFrame that read_sql should return.
    dummy_df = pd.DataFrame({
        "user_id": [1, 2],
        "vector_embedding": [b'data1', b'data2'],
        "location_lat": [10.0, 20.0],
        "location_lon": [30.0, 40.0],
        "charged_amount": [50.0, 60.0],
        "date": ["2023-01-01", "2023-01-02"]
    })

    # Monkeypatch pd.read_sql to return dummy_df.
    monkeypatch.setattr(pd, "read_sql", lambda statement, bind: dummy_df)
    
    # Create a mock Session.
    mock_db = MagicMock(spec=Session)
    mock_query = mock_db.query.return_value
    mock_query.statement = "dummy_statement"
    mock_db.bind = "dummy_bind"
    
    df = get_transactions_for_clustering(mock_db)
    # Ensure that the query was called and a DataFrame is returned.
    mock_db.query.assert_called_once()
    pd.testing.assert_frame_equal(df, dummy_df)

def test_parse_embedding_valid():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    arr_bytes = arr.tobytes()
    result = parse_embedding(arr_bytes)
    np.testing.assert_array_almost_equal(result, arr)

def test_parse_embedding_invalid():
    # Using a byte string with length not a multiple of 4.
    result = parse_embedding(b"abc")
    assert result.size == 0

def test_parse_transactions_df():
    # Build a sample dataframe with required columns.
    data = {
        "id": [1],
        "vector_embedding": [np.array([1.0, 2.0], dtype=np.float32).tobytes()],
        "location_lat": [10.0],
        "location_lon": [20.0],
        "charged_amount": [30.0],
        "date": ["2023-01-01"]
    }
    df = pd.DataFrame(data)
    df_parsed = parse_transactions_df(df)

    # Verify parsed_embedding exists and is a NumPy array.
    assert "parsed_embedding" in df_parsed.columns
    assert isinstance(df_parsed.loc[0, "parsed_embedding"], np.ndarray)

    # Verify the date is converted and date components are added.
    assert pd.api.types.is_datetime64_any_dtype(df_parsed["date"])
    for col in ["year", "month", "day"]:
        assert col in df_parsed.columns
    # Check expected date parts.
    assert df_parsed.loc[0, "year"] == 2023
    assert df_parsed.loc[0, "month"] == 1
    assert df_parsed.loc[0, "day"] == 1

def test_build_feature_matrix():
    # Create a DataFrame with two rows ensuring embeddings have the same length.
    data = {
        "id": [1, 2],
        "vector_embedding": [
            np.array([1.0, 2.0], dtype=np.float32).tobytes(),
            np.array([3.0, 4.0], dtype=np.float32).tobytes()
        ],
        "location_lat": [10.0, 20.0],
        "location_lon": [30.0, 40.0],
        "charged_amount": [50.0, 60.0],
        "date": ["2023-01-01", "2023-01-02"]
    }
    df = pd.DataFrame(data)
    df_parsed = parse_transactions_df(df)
    X, transaction_ids = build_feature_matrix(df_parsed)
    
    # Each row should be: embedding (2 floats) + 6 extra features = 8 elements.
    assert X.shape == (2, 8)
    assert transaction_ids == [1, 2]

def test_scale_features():
    # Create a simple feature matrix.
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    X_scaled = scale_features(X)
    
    # Check that each column has mean approximately 0 and std approximately 1.
    col_means = X_scaled.mean(axis=0)
    col_stds = X_scaled.std(axis=0)
    np.testing.assert_allclose(col_means, np.zeros_like(col_means), atol=1e-6)
    np.testing.assert_allclose(col_stds, np.ones_like(col_stds), atol=1e-6)