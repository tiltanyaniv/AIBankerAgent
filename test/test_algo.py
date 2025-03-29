import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session

# Ensure the project root is in sys.path if needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from your module.
from algo import (
    get_transactions_for_clustering,
    parse_embedding,
    parse_transactions_df,
    build_feature_matrix,
    scale_features,
    plot_k_distance,
    run_dbscan,
    analyze_dbscan,
    analyze_transactions_for_user,
    find_best_dbscan_params
)

# --- Existing fixtures ---

@pytest.fixture
def dummy_df():
    # Note: 'vector_embedding' bytes here may not be valid for np.frombuffer,
    # so these tests target functions that do not rely on proper embeddings.
    return pd.DataFrame({
        "id": [1, 2],
        "user_id": [1, 1],
        "vector_embedding": [b'data1', b'data2'],
        "location_lat": [10.0, 20.0],
        "location_lon": [30.0, 40.0],
        "charged_amount": [50.0, 60.0],
        "date": ["2023-01-01", "2023-01-02"]
    })

@pytest.fixture
def dummy_db(monkeypatch, dummy_df):
    db = MagicMock(spec=Session)
    query = db.query.return_value
    query.statement = "dummy_statement"
    db.bind = "dummy_bind"
    monkeypatch.setattr(pd, "read_sql", lambda statement, bind: dummy_df)
    return db

# A fixture with valid embeddings for functions that need proper numeric arrays.
@pytest.fixture
def dummy_valid_df():
    return pd.DataFrame({
        "id": [1, 2],
        "user_id": [1, 1],
        "vector_embedding": [
            np.array([1.0, 2.0], dtype=np.float32).tobytes(),
            np.array([3.0, 4.0], dtype=np.float32).tobytes()
        ],
        "location_lat": [10.0, 20.0],
        "location_lon": [30.0, 40.0],
        "charged_amount": [50.0, 60.0],
        "date": ["2023-01-01", "2023-01-02"]
    })

@pytest.fixture
def dummy_valid_db(monkeypatch, dummy_valid_df):
    db = MagicMock(spec=Session)
    query = db.query.return_value
    query.statement = "dummy_statement"
    db.bind = "dummy_bind"
    monkeypatch.setattr(pd, "read_sql", lambda statement, bind: dummy_valid_df)
    return db

# --- Existing Tests ---

def test_get_transactions_for_clustering(dummy_db, dummy_df):
    df = get_transactions_for_clustering(dummy_db, user_id=1)
    dummy_db.query.assert_called_once()
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
    assert "parsed_embedding" in df_parsed.columns
    assert isinstance(df_parsed.loc[0, "parsed_embedding"], np.ndarray)
    assert pd.api.types.is_datetime64_any_dtype(df_parsed["date"])
    for col in ["year", "month", "day"]:
        assert col in df_parsed.columns
    assert df_parsed.loc[0, "year"] == 2023
    assert df_parsed.loc[0, "month"] == 1
    assert df_parsed.loc[0, "day"] == 1

def test_build_feature_matrix():
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
    # Each row: embedding (2 floats) + 6 extra features = 8 elements.
    assert X.shape == (2, 8)
    assert transaction_ids == [1, 2]

def test_scale_features():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    X_scaled = scale_features(X)
    col_means = X_scaled.mean(axis=0)
    col_stds = X_scaled.std(axis=0)
    np.testing.assert_allclose(col_means, np.zeros_like(col_means), atol=1e-6)
    np.testing.assert_allclose(col_stds, np.ones_like(col_stds), atol=1e-6)

# --- Additional Tests for Improved Coverage ---

def test_plot_k_distance(monkeypatch):
    # Create a simple dataset.
    X = np.array([[0,0], [1,1], [2,2], [3,3], [4,4]], dtype=np.float32)
    # Override plt.show to prevent actual plot display.
    monkeypatch.setattr(plt, "show", lambda: None)
    kth_distances = plot_k_distance(X, k=1)
    # Check that kth_distances has the same number of points as X.
    assert len(kth_distances) == X.shape[0]
    # All distances should be non-negative.
    assert np.all(kth_distances >= 0)

def test_run_dbscan():
    # Create a dataset with two clear clusters.
    X_scaled = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=np.float32)
    labels = run_dbscan(X_scaled, eps=1.5, min_samples=2)
    assert isinstance(labels, np.ndarray)
    unique_labels = set(labels)
    # Expect at least two clusters (ignoring noise).
    assert len(unique_labels - {-1}) >= 2

def test_analyze_dbscan():
    labels = np.array([0, 0, -1, 1, -1])
    unique_labels, noise_count = analyze_dbscan(labels)
    assert unique_labels == {0, 1, -1}
    assert noise_count == 2

def test_find_best_dbscan_params_valid():
    # Create a dataset that can form two clusters.
    X_scaled = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=np.float32)
    eps_values = [1.5, 3.0]
    min_samples_values = [2, 3]
    best_params, best_silhouette, results = find_best_dbscan_params(X_scaled, eps_values, min_samples_values)
    # Should find a valid parameter combination.
    assert best_params is not None
    # Accept both built-in float and NumPy floating types.
    assert isinstance(best_silhouette, (float, np.floating))
    for key in results:
        assert isinstance(key, tuple)

def test_find_best_dbscan_params_invalid():
    # A dataset too small to compute silhouette score.
    X_scaled = np.array([[0, 0]], dtype=np.float32)
    eps_values = [0.5]
    min_samples_values = [2]
    best_params, best_silhouette, results = find_best_dbscan_params(X_scaled, eps_values, min_samples_values)
    assert best_params is None
    assert best_silhouette == -1

def test_analyze_transactions_for_user_no_transactions(monkeypatch):
    # Return an empty DataFrame to simulate no transactions.
    empty_df = pd.DataFrame(columns=["id", "user_id", "vector_embedding", "location_lat", "location_lon", "charged_amount", "date"])
    db = MagicMock(spec=Session)
    query = db.query.return_value
    query.statement = "dummy_statement"
    db.bind = "dummy_bind"
    monkeypatch.setattr(pd, "read_sql", lambda statement, bind: empty_df)
    result = analyze_transactions_for_user(db, user_id=1, grid_search=False)
    assert "message" in result
    assert "No transactions found" in result["message"]

def test_analyze_transactions_for_user_no_grid_search(monkeypatch, dummy_valid_df):
    # Test end-to-end analysis without grid search using valid embeddings.
    db = MagicMock(spec=Session)
    query = db.query.return_value
    query.statement = "dummy_statement"
    db.bind = "dummy_bind"
    monkeypatch.setattr(pd, "read_sql", lambda statement, bind: dummy_valid_df)
    result = analyze_transactions_for_user(db, user_id=1, grid_search=False, default_eps=0.5, default_min_samples=2)
    expected_keys = {"user_id", "noise_count", "anomalous_transactions", "anomalous_transaction_ids", "eps_used", "min_samples_used"}
    assert expected_keys.issubset(result.keys())

def test_analyze_transactions_for_user_with_grid_search(monkeypatch, dummy_valid_df):
    # Test end-to-end analysis with grid search.
    db = MagicMock(spec=Session)
    query = db.query.return_value
    query.statement = "dummy_statement"
    db.bind = "dummy_bind"
    monkeypatch.setattr(pd, "read_sql", lambda statement, bind: dummy_valid_df)
    # Provide a narrow parameter range so grid search executes.
    result = analyze_transactions_for_user(
        db,
        user_id=1,
        grid_search=True,
        eps_range=np.array([0.1, 0.5]),
        min_samples_range=[1, 2],
        default_eps=0.3,
        default_min_samples=1
    )
    expected_keys = {"user_id", "noise_count", "anomalous_transactions", "anomalous_transaction_ids", "eps_used", "min_samples_used"}
    assert expected_keys.issubset(result.keys())