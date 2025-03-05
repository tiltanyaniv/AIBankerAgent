import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from models import Transaction  # Assumes Transaction model is defined appropriately
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Step 1: Query the required data from the database.
def get_transactions_for_clustering(db: Session) -> pd.DataFrame:
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
        Transaction.user_id,
        Transaction.vector_embedding,
        Transaction.location_lat,
        Transaction.location_lon,
        Transaction.charged_amount,
        Transaction.date
    )
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
      - Converts date to datetime and extracts year, month, day, hour, and weekday
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
    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.weekday  # 0=Monday, 6=Sunday
    
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

