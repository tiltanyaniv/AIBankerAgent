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
from forex_python.converter import CurrencyRates



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
    Reads transactions from JSON, extracts store locations, generates embeddings,
    and saves them to the database with latitude & longitude.
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

                # Get location and coordinates
                location_data = get_location(description)
                location = location_data
                location_lat, location_lon = get_location_coordinates(location)

                # Generate a transaction vector embedding
                vector_embedding = create_transaction_vector(original_currency, description, category).tobytes()

                new_transaction = Transaction(
                    user_id=user.id,
                    charged_amount=convert_to_usd(transaction["chargedAmount"], original_currency),
                    description=description,
                    category=category,
                    date=transaction_date,
                    original_currency=original_currency,
                    location=location,
                    location_lat=location_lat,
                    location_lon=location_lon,
                    vector_embedding=vector_embedding
                )

                db.add(new_transaction)

        db.commit()
        return {"status": "success", "message": "Transactions saved with location coordinates."}

    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}       

 # Load OpenAI API Key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI

def get_location(description):
    """
    It asks OpenAI if the transaction was online or physical.
    If OpenAI says it's not online, then we try to extract a location.
    """

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that analyzes transaction descriptions and determines if the purchase was made online or in a physical location."},
                {"role": "user", "content": f"Was the following purchase made online or in a physical store? Description: '{description}' Please respond with either 'Online Purchase' or 'Physical Store'."}
            ]
        )
        result = response.choices[0].message.content.strip()

        if result == "Online Purchase":
            return "Online Purchase"
        elif result == "Physical Store":
            # After determining it's a physical store, we can follow up with location detection
            location_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an assistant that extracts store locations from transaction descriptions. Only return the city name or the country name (whichever is available). No explanations, just the location."},
                    {"role": "user", "content": f"Where is the store location for this transaction: '{description}'?"}
                ]
            )
            return location_response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error detecting location: {e}")
        return "Unknown"
    

def get_location_coordinates(location_str: str):
    """
    Fetch latitude and longitude for a given location string using Nominatim.
    Returns a tuple (latitude, longitude) as floats, or (None, None) if not found.
    """
    geolocator = Nominatim(user_agent="AIBankerAgent")
    try:
        geo_location = geolocator.geocode(location_str)
        if geo_location:
            lat = float(geo_location.latitude)
            lon = float(geo_location.longitude)
            return lat, lon
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
        model="text-embedding-3-large",  # Use the latest embedding model
        input=[text],  # Ensure input is a list
        encoding_format="float"  # Specify encoding format to ensure numerical output
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def create_transaction_vector(original_currency, description, category):
    """
    Creates a vector representation combining all relevant transaction details.
    """
    text_data = f"{original_currency}, {description}, {category}"
    return get_embedding(text_data)


def convert_to_usd(amount: float, original_currency: str) -> float:
    """
    Converts an amount from the original currency to US dollars.
    
    Args:
        amount (float): The amount in the original currency.
        original_currency (str): The 3-letter ISO code for the original currency (e.g., "EUR", "GBP").
    
    Returns:
        float: The equivalent amount in US dollars.
    """
    c = CurrencyRates()
    try:
        # If the original currency is already USD, return the amount as is.
        if original_currency.upper() == "USD":
            return amount
        # Get the conversion rate from the original currency to USD.
        rate = c.get_rate(original_currency.upper(), "USD")
        return amount * rate
    except Exception as e:
        print(f"Error converting currency from {original_currency} to USD: {e}")
        # Optionally, you could raise the exception or return the original amount.
        return amount