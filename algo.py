import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from models import Transaction  # Assumes Transaction model is defined appropriately
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

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

def analyze_transactions_for_user(db: Session, user_id: int, grid_search: bool = True,
                                  eps_range: np.ndarray = None, min_samples_range: list = None,
                                  default_eps: float = 0.5, default_min_samples: int = 5):
    """
    Analyzes transactions for a specific user using DBSCAN clustering.
    If grid_search is True, it will first search for the best eps and min_samples
    parameters (using a silhouette score) and then run DBSCAN with those parameters.
    Returns only the anomalous transactions (noise points, label == -1).

    Parameters:
      db (Session): SQLAlchemy session.
      user_id (int): The user id whose transactions are to be analyzed.
      grid_search (bool): Whether to run grid search for best parameters.
      eps_range (np.ndarray): Optional range of eps values to try (default: np.linspace(50, 80, 11)).
      min_samples_range (list): Optional list of min_samples values to try (default: [2, 3, 4]).
      default_eps (float): Fallback eps value if grid search finds no valid parameters.
      default_min_samples (int): Fallback min_samples value if grid search finds no valid parameters.
      
    Returns:
      dict: A dictionary containing:
            - user_id,
            - the eps and min_samples used,
            - noise_count,
            - anomalous_transactions (details),
            - anomalous_transaction_ids,
            - grid_search_results (all combinations evaluated).
    """
    # Get transactions for the specified user.
    df = get_transactions_for_clustering(db, user_id)
    if df.empty:
        return {"message": f"No transactions found for user {user_id}"}
    
    # Parse the DataFrame (convert embeddings, date fields, etc.).
    df = parse_transactions_df(df)
    
    # Build the feature matrix and retrieve transaction IDs.
    X, transaction_ids = build_feature_matrix(df)
    
    # Scale the features.
    X_scaled = scale_features(X)
    
    # Determine DBSCAN parameters.
    if grid_search:
        if eps_range is None:
            eps_range = np.linspace(70, 100, 11)
        if min_samples_range is None:
            min_samples_range = [2, 7, 12]
        
        best_params, best_score, all_results = find_best_dbscan_params(X_scaled, eps_range, min_samples_range)
        if best_params is None:
            eps, min_samples = default_eps, default_min_samples
            grid_search_results = None
        else:
            eps, min_samples = best_params
            grid_search_results = {str(k): v for k, v in all_results.items()}
    else:
        eps, min_samples = default_eps, default_min_samples
        grid_search_results = None
    
    # Run DBSCAN with the chosen parameters.
    labels = run_dbscan(X_scaled, eps=eps, min_samples=min_samples)
    
    # Filter transactions that are anomalies (label == -1).
    anomaly_indices = [i for i, lab in enumerate(labels) if lab == -1]
    anomalous_transactions = df.iloc[anomaly_indices].copy()
    anomalous_transactions.drop(columns=["vector_embedding", "parsed_embedding"], inplace=True)
    anomalies_list = anomalous_transactions.to_dict(orient="records")
    anomaly_ids = [transaction_ids[i] for i in anomaly_indices]
    
    result = {
        "user_id": user_id,
         "noise_count": int(sum(1 for lab in labels if lab == -1)),
         "anomalous_transactions": anomalies_list,
         "anomalous_transaction_ids": anomaly_ids,
         "eps_used": eps,
         "min_samples_used": min_samples,
    }
    
    return result

def find_best_dbscan_params(X_scaled, eps_values, min_samples_values):
    """
    Finds the best eps and min_samples for DBSCAN based on the silhouette score.
    
    Parameters:
      X_scaled (numpy.ndarray): The scaled feature matrix.
      eps_values (list or numpy.ndarray): A range of eps values to try.
      min_samples_values (list or numpy.ndarray): A range of min_samples values to try.
    
    Returns:
      best_params (tuple): The best (eps, min_samples) combination.
      best_silhouette (float): The silhouette score for the best parameters.
      results (dict): Dictionary of all parameter combinations with their silhouette scores.
    """
    best_params = None
    best_silhouette = -1  # silhouette score ranges from -1 to 1
    results = {}
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_scaled)
            
            # Filter out noise points for silhouette score calculation.
            core_mask = labels != -1
            
            # If there is only one cluster or no points after filtering, skip this combination.
            if len(set(labels)) <= 1 or np.sum(core_mask) < 2:
                results[(eps, min_samples)] = None
                continue
            
            try:
                score = silhouette_score(X_scaled[core_mask], labels[core_mask])
                results[(eps, min_samples)] = score
                if score > best_silhouette:
                    best_silhouette = score
                    best_params = (eps, min_samples)
            except Exception as e:
                results[(eps, min_samples)] = None
                continue
                
    return best_params, best_silhouette, results
