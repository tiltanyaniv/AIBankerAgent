import json
import os
import requests
import numpy as np
import openai
from sqlalchemy.orm import Session
from models import User, Transaction
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from geopy.geocoders import Nominatim
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sqlalchemy import text
<<<<<<< HEAD
=======

>>>>>>> a45b007 (Data Extraction and Parsing for DBscan)

# Load OpenAI API Key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI


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
    and saves them to the database with latitude & longitude (and USD-converted amounts).
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
                # Get location and coordinates
                location_data = get_location(description)
                location = location_data
                location_lat, location_lon = get_location_coordinates(location)

                # Generate a transaction vector embedding
                vector_embedding = create_transaction_vector(original_currency, description, category).tobytes()

                # Convert the original amount to USD
                usd_amount = convert_to_usd(transaction["chargedAmount"], original_currency)

                new_transaction = Transaction(
                    user_id=user.id,
                    charged_amount=usd_amount,
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
        return {"status": "success", "message": "Transactions saved with location coordinates and USD amounts."}

    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}


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
                    {"role": "system", "content": "You are an assistant that extracts store locations from transaction descriptions.Ignore business names and return only the city or country. Only return the city name or the country name (whichever is available). No explanations, just the location."},
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
    Returns a tuple (latitude, longitude) as floats.
    If the location is 'Online Purchase' or unknown, returns default coordinates.
    """
    # Define default coordinates for online/unknown locations
    default_coords = (0.0, 0.0)  # or use (lat, lon) for a default country center

    # If the location string is empty or clearly indicates an online transaction, return defaults
    if not location_str or location_str.strip().lower() in ["Unknown", "online purchase"]:
        return default_coords

    geolocator = Nominatim(user_agent="AIBankerAgent")
    try:
        geo_location = geolocator.geocode(location_str)
        if geo_location:
            lat = float(geo_location.latitude)
            lon = float(geo_location.longitude)
            return lat, lon
        else:
            return default_coords
    except Exception as e:
        print(f"Error retrieving geocode for {location_str}: {e}")
        return default_coords


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
    Converts an amount from the original currency to US dollars using exchangerate-api.com.
    Always uses ILS as the base and returns the converted amount without printing debug info.
    """
    if original_currency.upper() == "USD":
        return amount

    # Build the URL using the original currency as the base.
    url = f"https://api.exchangerate-api.com/v4/latest/{original_currency.upper()}"
    try:
        response = requests.get(url)
        data = response.json()
        # Remove or comment out the debug print:
        # print("DEBUG: exchangerate-api.com response:", data)
        
        rates = data.get("rates", {})
        rate_usd = rates.get("USD")
        if rate_usd is None:
            raise Exception("Conversion rate not found for USD.")
        
        return amount * rate_usd

    except Exception as e:
<<<<<<< HEAD
        return amount


=======
        # Optionally, log the error using a logging framework if needed.
        # For now, we'll just return the original amount.
        # print(f"Error converting currency from {original_currency} to USD: {e}")
        return amount
>>>>>>> a45b007 (Data Extraction and Parsing for DBscan)
