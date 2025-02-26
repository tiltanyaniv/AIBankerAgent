import json
import numpy as np
import openai
import os
from sqlalchemy.orm import Session
from models import User, Transaction
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from geopy.geocoders import Nominatim



def create_user(db: Session, account_number: str):
    """Create a new user if they don't exist."""
    existing_user = db.query(User).filter(User.username == account_number).first()
    if existing_user:
        return existing_user  # Prevent duplicate users

    user = User(username=account_number)  # Store accountNumber as username
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def save_transactions_from_json(db: Session, file_path: str = "transactions.json"):
    """
    Reads transactions from JSON, extracts store locations, generates embeddings, and saves them to the database.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for account in data:
            account_number = str(account.get("accountNumber")).strip()

            # Ensure user exists or create them
            user = db.query(User).filter(User.username == account_number).first()
            if not user:
                user = User(username=account_number)
                db.add(user)
                db.commit()
                db.refresh(user)

            for transaction in account.get("transactions", []):
                transaction_date = datetime.strptime(transaction["date"], "%Y-%m-%dT%H:%M:%S.%fZ")
                description = transaction["description"]
                original_currency = transaction["originalCurrency"]
                category = transaction["category"]

                # Extract store location using OpenAI
                location = get_location(description)

                # Generate a transaction vector embedding
                vector_embedding = create_transaction_vector(original_currency, description, category, location).tobytes()

                new_transaction = Transaction(
                    user_id=user.id,
                    charged_amount=transaction["chargedAmount"],
                    description=description,
                    category=category,
                    date=transaction_date,
                    original_currency=original_currency,
                    location=location,
                    vector_embedding=vector_embedding
                )

                db.add(new_transaction)

        db.commit()
        return {"status": "success", "message": "Transactions saved with location and embeddings."}

    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}

# Load OpenAI API Key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI

def get_location(description):
    """
    Uses OpenAI GPT-4o to extract the store location from the transaction description.
    """
    try:
        client = OpenAI()  # Correct OpenAI API client initialization
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the correct model name
            messages=[
                {"role": "system", "content": "You are an assistant that extracts store locations from transaction descriptions. Only return the city name or the country name, nothing else."},
                {"role": "user", "content": f"Where is the store location for this transaction: '{description}'? Provide only the city name or the country name. Do not include any extra words or explanations."}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Unknown"
    

def get_location_coordinates(location_str: str):
    """
    Fetch latitude and longitude for a given location string using Nominatim.
    Returns a tuple (latitude, longitude) or (None, None) if not found.
    """
    geolocator = Nominatim(user_agent="AIBankerAgent")
    try:
        geo_location = geolocator.geocode(location_str)
        if geo_location:
            return geo_location.latitude, geo_location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Error retrieving geocode for {location_str}: {e}")
        return None, None

def get_embedding(text):
    """
    Generate an embedding for a given text using OpenAI's updated API.
    """
    client = openai.OpenAI()  # The new API requires creating a client instance
    response = client.embeddings.create(
        model="text-embedding-3-small",  # Use the latest embedding model
        input=[text],  # Ensure input is a list
        encoding_format="float"  # Specify encoding format to ensure numerical output
    )
    return np.array(response.data[0].embedding, dtype=np.float32)
def create_transaction_vector(original_currency, description, category, location):
    """
    Creates a vector representation combining all relevant transaction details.
    """
    text_data = f"{original_currency}, {description}, {category}, {location}"
    return get_embedding(text_data)

def detect_unusual_transactions(db: Session, threshold=0.2):
    """
    Detects transactions with unusual vectors using OpenAI embeddings.
    """
    transactions = db.query(Transaction).filter(Transaction.vector_embedding.isnot(None)).all()

    if len(transactions) < 2:
        return {"message": "Not enough transactions to analyze."}

    # Convert stored embeddings back to numpy arrays
    embeddings = np.array([np.frombuffer(t.vector_embedding, dtype=np.float32) for t in transactions])

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Find the lowest similarity for each transaction
    min_similarities = np.min(similarity_matrix + np.eye(len(transactions)), axis=1)

    # Flag transactions with low similarity scores
    unusual_transactions = [
        {
            "id": transactions[i].id,
            "description": transactions[i].description,
            "category": transactions[i].category,
            "location": transactions[i].location,
            "charged_amount": transactions[i].charged_amount,
            "similarity_score": float(min_similarities[i])
        }
        for i in range(len(transactions)) if min_similarities[i] < threshold
    ]

    return {
        "total_transactions": len(transactions),
        "unusual_transactions": unusual_transactions
    }