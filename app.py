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
    Updates the BANK_USERNAME and BANK_PASSWORD values in index.js.
    """
    try:
        file_path = "index.js"  # Adjust this if your file is in a different location
        with open(file_path, "r") as f:
            content = f.read()

        # Update the BANK_USERNAME variable
        content, count_username = re.subn(
            r'(let\s+BANK_USERNAME\s*=\s*")[^"]*(")',
            r'\1' + credentials.username + r'\2',
            content
        )

        # Update the BANK_PASSWORD variable
        content, count_password = re.subn(
            r'(let\s+BANK_PASSWORD\s*=\s*")[^"]*(")',
            r'\1' + credentials.password + r'\2',
            content
        )

        with open(file_path, "w") as f:
            f.write(content)

        return {"status": "success", "message": "Credentials updated successfully in index.js."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-credentials")
def get_credentials():
    """
    Retrieves BANK_USERNAME and BANK_PASSWORD from the environment variables.
    """
    try:
        username = os.getenv("BANK_USERNAME", "Not set")
        password = os.getenv("BANK_PASSWORD", "Not set")

        return {"BANK_USERNAME": username, "BANK_PASSWORD": password}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/get-transData")
def get_transData():
    """
    Retrieves companyId and startDate from index.js.
    """
    try:
        file_path = "index.js"  # Adjust the path if necessary
        with open(file_path, "r") as f:
            content = f.read()

        company_id_match = re.search(r"companyId:\s*CompanyTypes\.([A-Za-z0-9_]+)", content)
        start_date_match = re.search(r"startDate:\s*new Date\('([^']+)'\)", content)

        company_id = company_id_match.group(1) if company_id_match else "Not found"
        start_date = start_date_match.group(1) if start_date_match else "Not found"

        return {"company_id": company_id, "start_date": start_date}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
class TransData(BaseModel):
    company_id: str   
    start_date: str   

@app.post("/set-transData")
def set_transData(data: TransData):
    """
    Updates the companyId and startDate in index.js with values provided by the user.
    """
    try:
        file_path = "index.js"  # Adjust the path if necessary
        with open(file_path, "r") as f:
            content = f.read()

        content = re.sub(
            r"companyId:\s*CompanyTypes\.[A-Za-z0-9_]+",
            f"companyId: CompanyTypes.{data.company_id}",
            content
        )
        
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
    Runs the index.js script using Node.js and provides detailed logs if something goes wrong.
    """
    try:
        # Run index.js using Node.js
        result = subprocess.run(
            ["node", "index.js"],
            capture_output=True,
            text=True
        )

        # Always print stdout and stderr to the server logs for debugging
        print("=== run_transaction: STDOUT ===")
        print(result.stdout)
        print("=== run_transaction: STDERR ===")
        print(result.stderr)

        # If Node.js returns a non-zero exit code, raise an error but still return logs
        if result.returncode != 0:
            error_details = {
                "returncode": result.returncode,
                "cmd": result.args,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            print("Detailed error information:", error_details)
            raise HTTPException(status_code=500, detail=f"Error running index.js: {error_details}")

        # If everything is OK, return the logs
        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except Exception as e:
        # Log any unexpected errors
        print("An unexpected error occurred:", e)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
@app.get("/analyze-transactions/{user_id}")
def analyze_transactions(user_id: int, grid_search: bool = True, eps: float = 0.5, min_samples: int = 5, db: Session = Depends(get_db)):
    try:
        result = algo.analyze_transactions_for_user(db, user_id, grid_search=grid_search, default_eps=eps, default_min_samples=min_samples)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize-transactions/{user_id}")
def visualize_transactions(
    user_id: int,
    grid_search: bool = True,
    eps: float = 0.5,
    min_samples: int = 5,
    db: Session = Depends(get_db)
):
    """
    Visualizes the clustering of transactions for a given user by:
      1. Running analyze_transactions_for_user to automatically determine the best DBSCAN parameters.
      2. Reprocessing the user's transactions, running DBSCAN with those parameters, and reducing dimensions via PCA.
      3. Returning a scatter plot (PNG image) of the clusters.
    """
    try:
        # Run the analysis function to get the best parameters.
        analysis_result = algo.analyze_transactions_for_user(
            db,
            user_id,
            grid_search=grid_search,
            default_eps=eps,
            default_min_samples=min_samples
        )
        
        # Extract the best parameters from the analysis result.
        best_eps = analysis_result.get("eps_used", eps)
        best_min_samples = analysis_result.get("min_samples_used", min_samples)
        
        # Retrieve and process the user's transactions.
        df = algo.get_transactions_for_clustering(db, user_id)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No transactions found for user {user_id}")
        df = algo.parse_transactions_df(df)
        X, transaction_ids = algo.build_feature_matrix(df)
        X_scaled = algo.scale_features(X)
        
        # Run DBSCAN with the best parameters.
        labels = algo.run_dbscan(X_scaled, eps=best_eps, min_samples=best_min_samples)
        
        # Use PCA to reduce the feature space to 2 dimensions.
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Generate a scatter plot.
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.8)
        plt.title(f"Transaction Clusters for User {user_id}\n(eps={best_eps:.2f}, min_samples={best_min_samples})")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster Label")
        
        # Save plot to a BytesIO buffer.
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()  # Clean up the plot
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))