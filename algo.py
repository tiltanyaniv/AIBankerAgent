import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from models import Transaction  # Assumes Transaction model is defined appropriately
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# Step 1: Query the required data from the database.
def get_transactions_for_clustering(db: Session, user_id: int) -> pd.DataFrame:
    """
    Queries the database for transactions needed for DBSCAN.
    Selects the following columns:
      - user_id (to identify transaction owner)
      - vector_embedding
      - location_lat, location_lon
      - charged_amount (in USD)
      - date
    Returns a Pandas DataFrame.
    """
    query = db.query(
        Transaction.id,
        Transaction.user_id,
        Transaction.vector_embedding,
        Transaction.location_lat,
        Transaction.location_lon,
        Transaction.charged_amount,
        Transaction.date
    ).filter(Transaction.user_id == user_id)
    # Using the query's statement and the DB's engine binding to load into a DataFrame
    df = pd.read_sql(query.statement, db.bind)
    return df

# Step 2: Parse each field from the DataFrame.
def parse_embedding(embedding_data) -> np.ndarray:
    """
    Converts the stored embedding (assumed to be in bytes) into a NumPy array.
    Adjust the dtype and conversion if your storage format is different.
    """
    try:
        # Assuming the embedding is stored as bytes and was created using np.array(...).tobytes()
        arr = np.frombuffer(embedding_data, dtype=np.float32)
        return arr
    except Exception as e:
        print(f"Error parsing embedding: {e}")
        return np.array([])

def parse_transactions_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the DataFrame fields so that they are ready for vector creation:
      - Converts vector_embedding to a NumPy array (new column: parsed_embedding)
      - Ensures location_lat and location_lon are floats
      - Converts charged_amount to float
      - Converts date to datetime and extracts year, month, and day
    Returns the updated DataFrame.
    """
    # Parse vector_embedding column into a NumPy array.
    df["parsed_embedding"] = df["vector_embedding"].apply(parse_embedding)
    
    # Ensure location columns are floats.
    df["location_lat"] = df["location_lat"].astype(float)
    df["location_lon"] = df["location_lon"].astype(float)
    
    # Ensure charged_amount is float.
    df["charged_amount"] = df["charged_amount"].astype(float)
    
    # Convert date string to datetime object.
    df["date"] = pd.to_datetime(df["date"])
    
    # Extract numeric features from the date.
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    
    return df

def build_feature_matrix(df: pd.DataFrame):
    """
    Combines various features from the transactions DataFrame into a single feature matrix.
    
    Expected columns in df:
      - 'parsed_embedding': a NumPy array representing the transaction embedding.
      - 'location_lat': float (latitude)
      - 'location_lon': float (longitude)
      - 'charged_amount': float
      - 'id': a transaction identifier (optional, to track which row corresponds to which transaction)
    
    This function concatenates the embedding with the numeric features (coordinates and amount).
    You can modify this to include additional features (like date components) if desired.
    """
    if df.empty:
        return np.array([]), []

    feature_list = []
    transaction_ids = []  # to track transaction IDs

    for _, row in df.iterrows():
        emb = row['parsed_embedding']
        # Get additional numeric features.
        # Use 0.0 if the coordinate is missing.
        lat = row['location_lat'] if pd.notnull(row['location_lat']) else 0.0
        lon = row['location_lon'] if pd.notnull(row['location_lon']) else 0.0
        amount = row['charged_amount']
        # Date features
        year = row['year']
        month = row['month']
        day = row['day']


        # Concatenate all these features into one vector.
        # You might decide on the order: embedding first, then the additional features.
        extra_features = np.array([lat, lon, amount, year, month, day], dtype=np.float32)
        combined = np.concatenate([emb, extra_features])
        feature_list.append(combined)
        transaction_ids.append(row['id'])


    # Stack all vectors into a matrix.
    X = np.vstack(feature_list)
    return X, transaction_ids

def scale_features(X: np.array):
    """
    Scales the feature matrix X using StandardScaler.
    
    Returns the scaled feature matrix.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled



 
def plot_k_distance(X, k=5):
    """
    Computes and plots the k-distance graph (distance to the kth nearest neighbor) for each point in X.
    
    Parameters:
      X (numpy.ndarray): Feature matrix.
      k (int): Which nearest neighbor distance to use (e.g., 5 means the 5th nearest neighbor, excluding self).
      
    Returns:
      kth_distances (numpy.ndarray): Sorted distances of the kth neighbor for each sample.
    """
    # When using NearestNeighbors, the first neighbor is the point itself (distance 0).
    nbrs = NearestNeighbors(n_neighbors=k+1)  # k+1 because self is included
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    # The kth nearest neighbor (excluding self) is at index k.
    kth_distances = np.sort(distances[:, k])
    
    plt.figure(figsize=(10, 6))
    plt.plot(kth_distances, marker="o", linestyle="--")
    plt.xlabel("Samples sorted by kth neighbor distance")
    plt.ylabel(f"Distance to {k}th nearest neighbor")
    plt.title("K-Distance Graph for DBSCAN eps selection")
    plt.show()
    
    return kth_distances

def run_dbscan(X_scaled, eps, min_samples):
    """
    Runs DBSCAN clustering on the scaled feature matrix.
    
    Parameters:
      X_scaled (numpy.ndarray): Scaled feature matrix.
      eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
      min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
      labels (numpy.ndarray): Cluster labels for each sample.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    return labels

def analyze_dbscan(labels):
    """
    Prints the cluster labels and number of noise points.
    
    Parameters:
      labels (numpy.ndarray): Cluster labels returned by DBSCAN.
      
    Returns:
      unique_labels (set): The unique cluster labels.
      noise_count (int): The count of noise points (where label == -1).
    """
    unique_labels = set(labels)
    noise_count = np.sum(labels == -1)
    return unique_labels, noise_count

def analyze_transactions_for_user(db: Session, user_id: int, eps: float = 0.5, min_samples: int = 5):
    """
    Analyzes transactions for a specific user using DBSCAN clustering and returns only the anomalous transactions.
    
    Parameters:
      db (Session): SQLAlchemy session.
      user_id (int): The ID of the user whose transactions will be analyzed.
      eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
      min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
      dict: A dictionary containing the user_id, count of noise points, and a list of anomalous transaction details.
    """
    # Step 1: Get transactions for the specified user.
    df = get_transactions_for_clustering(db, user_id)
    if df.empty:
        return {"message": f"No transactions found for user {user_id}"}
    
    # Step 2: Parse the DataFrame (convert embeddings, date fields, etc.).
    df = parse_transactions_df(df)
    
    # Step 3: Build the feature matrix and retrieve transaction IDs.
    X, transaction_ids = build_feature_matrix(df)
    
    # Step 4: Scale the features.
    X_scaled = scale_features(X)
    
    # Step 5: Run DBSCAN clustering.
    labels = run_dbscan(X_scaled, eps=eps, min_samples=min_samples)
    
    # Step 6: Filter transactions that are anomalies (label == -1).
    anomaly_indices = [i for i, lab in enumerate(labels) if lab == -1]
    anomalous_transactions = df.iloc[anomaly_indices]
    anomalous_transactions.drop(columns=["vector_embedding", "parsed_embedding"], inplace=True)
    # Optionally, convert any numpy data types in the anomalies for JSON serialization.
    anomalies_list = anomalous_transactions.to_dict(orient="records")

    anomaly_ids = [transaction_ids[i] for i in anomaly_indices]
    
    return {
        "user_id": user_id,
        "noise_count": int(sum(1 for lab in labels if lab == -1)),
        "anomalous_transactions": anomalies_list,
        "anomalous_transaction_ids": anomaly_ids
    }
