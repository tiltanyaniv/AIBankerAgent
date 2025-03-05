import numpy as np
import pandas as pd
import pytest
from algo import build_feature_matrix, scale_features

def test_build_feature_matrix_empty():
    # Create an empty DataFrame with the expected columns.
    df = pd.DataFrame(columns=[
        'parsed_embedding', 'location_lat', 'location_lon', 
        'charged_amount', 'year', 'month', 'day', 'id'
    ])
    X, ids = build_feature_matrix(df)
    # When DataFrame is empty, X should be an empty array and ids an empty list.
    assert X.size == 0
    assert ids == []

def test_build_feature_matrix_single_row():
    # Create a dummy row with known values.
    emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    data = {
        'parsed_embedding': [emb],
        'location_lat': [40.0],
        'location_lon': [-74.0],
        'charged_amount': [100.0],
        'year': [2025],
        'month': [3],
        'day': [15],
        'id': [1]
    }
    df = pd.DataFrame(data)
    X, ids = build_feature_matrix(df)
    
    # Expect the combined vector to be embedding (3) + extra features (6) = 9 dimensions.
    assert X.shape == (1, 9)
    
    # Check that the first 3 values match the embedding.
    np.testing.assert_array_equal(X[0, :3], emb)
    # The extra features should be [lat, lon, amount, year, month, day].
    expected_extra = np.array([40.0, -74.0, 100.0, 2025, 3, 15], dtype=np.float32)
    np.testing.assert_array_equal(X[0, 3:], expected_extra)
    
    # The transaction_ids list should contain the id.
    assert ids == [1]

def test_build_feature_matrix_multiple_rows():
    # Create two dummy rows.
    emb1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    emb2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    data = {
        'parsed_embedding': [emb1, emb2],
        'location_lat': [40.0, 41.0],
        'location_lon': [-74.0, -73.0],
        'charged_amount': [100.0, 200.0],
        'year': [2025, 2024],
        'month': [3, 4],
        'day': [15, 16],
        'id': [1, 2]
    }
    df = pd.DataFrame(data)
    X, ids = build_feature_matrix(df)
    
    # Expect shape (2, 9) since each row produces a vector of 9 dimensions.
    assert X.shape == (2, 9)
    # Check that the transaction IDs are correct.
    assert ids == [1, 2]

def test_scale_features():
    # Create a simple 2x3 matrix.
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=np.float32)
    X_scaled = scale_features(X)
    
    # The shape should remain the same.
    assert X_scaled.shape == X.shape
    
    # Check that each column has mean ~0 and std ~1.
    means = X_scaled.mean(axis=0)
    stds = X_scaled.std(axis=0)
    np.testing.assert_allclose(means, np.zeros_like(means), atol=1e-6)
    np.testing.assert_allclose(stds, np.ones_like(stds), atol=1e-6)