from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import crud
import algo
from pydantic import BaseModel
import datetime
import os
import pandas as pd
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
from sklearn.decomposition import PCA
from database import SessionLocal
import re
import subprocess


# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models
class UserCreate(BaseModel):
    username: str

class TransactionCreate(BaseModel):
    user_id: int
    charged_amount: float
    description: str
    category: str
    date: datetime.datetime
    original_currency: str

@app.post("/users/")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db, user.username)


class Credentials(BaseModel):
    username: str
    password: str

@app.post("/load-transactions/")
def load_transactions(db: Session = Depends(get_db)):
    """Reads transactions from JSON and saves them to the database."""
    result = crud.save_transactions_from_json(db)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result


@app.post("/set-credentials")
def set_credentials(credentials: Credentials):
    """
    Save BANK_USERNAME and BANK_PASSWORD to the .env file.
    """
    try:
        env_file_path = ".env"
        with open(env_file_path, "w") as env_file:
            env_file.write(f"BANK_USERNAME={credentials.username}\n")
            env_file.write(f"BANK_PASSWORD={credentials.password}\n")

        os.environ["BANK_USERNAME"] = credentials.username
        os.environ["BANK_PASSWORD"] = credentials.password

        return {"status": "success", "message": "Credentials saved successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


class TransData(BaseModel):
    company_id: str   # e.g., "otsarHahayal"
    start_date: str   # e.g., "2025-01-01"

@app.post("/set-transData")
def set_transData(data: TransData):
    """
    Updates the companyId and startDate in index.js with values provided by the user.
    """
    try:
        file_path = "index.js"  # Adjust the path if necessary
        with open(file_path, "r") as f:
            content = f.read()

        # Replace the companyId line (e.g., "companyId: CompanyTypes.otsarHahayal")
        content = re.sub(
            r"companyId:\s*CompanyTypes\.[A-Za-z0-9_]+",
            f"companyId: CompanyTypes.{data.company_id}",
            content
        )
        
        # Replace the startDate line (e.g., "startDate: new Date('2025-01-01')")
        content = re.sub(
            r"startDate:\s*new Date\('[^']+'\)",
            f"startDate: new Date('{data.start_date}')",
            content
        )

        with open(file_path, "w") as f:
            f.write(content)

        return {"status": "success", "message": "Transaction data updated successfully in index.js."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.post("/run-transaction")
def run_transaction():
    """
    Runs the index.js script using Node.js.
    """
    try:
        # Run index.js using Node.js
        result = subprocess.run(
            ["node", "index.js"],
            capture_output=True,
            text=True,
            check=True
        )
        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error running index.js: {e.stderr}")

@app.get("/analyze-transactions/{user_id}")
def analyze_transactions(user_id: int, grid_search: bool = True, eps: float = 0.5, min_samples: int = 5, db: Session = Depends(get_db)):
    try:
        result = algo.analyze_transactions_for_user(db, user_id, grid_search=grid_search, default_eps=eps, default_min_samples=min_samples)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize-transactions/{user_id}")
def visualize_transactions(user_id: int, eps: float = 0.5, min_samples: int = 5, db: Session = Depends(get_db)):
    """
    Generates a 2D scatter plot of the transactions for a specific user.
    It builds the feature matrix, scales it, reduces dimensionality with PCA,
    runs DBSCAN to get cluster labels, and returns the plot as an image.
    """
    try:
        # Query transactions for the user and parse the DataFrame.
        df = algo.get_transactions_for_clustering(db, user_id)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No transactions found for user {user_id}")
        df = algo.parse_transactions_df(df)  # Ensure this function converts raw embeddings and extracts date parts
        
        # Build feature matrix and scale it.
        X, transaction_ids = algo.build_feature_matrix(df)
        X_scaled = algo.scale_features(X)
        
        # Run DBSCAN to get cluster labels.
        labels = algo.run_dbscan(X_scaled, eps=eps, min_samples=min_samples)
        
        # Reduce dimensionality to 2D using PCA.
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create scatter plot.
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.8)
        plt.title(f"Transaction Clusters for User {user_id}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster Label")
        
        # Save the plot to a BytesIO buffer.
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()  # Close the figure to free memory
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))