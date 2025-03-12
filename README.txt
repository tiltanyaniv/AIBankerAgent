# Bank Account Anomaly Detection

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
  - [Python Dependencies](#python-dependencies)
  - [Node.js Dependencies](#nodejs-dependencies)
  - [Database Requirements](#database-requirements)
- [Setup and Installation](#setup-and-installation)
  - [Clone the Repository](#clone-the-repository)
  - [Install Python Dependencies](#install-python-dependencies)
  - [Install Node.js Dependencies](#install-nodejs-dependencies)
  - [Database Setup](#database-setup)
  - [Start the Server](#start-the-server)
  - [Windows-Specific Instructions](#windows-specific-instructions)
  - [Mac-Specific Instructions](#mac-specific-instructions)
- [Usage Instructions](#usage-instructions)
- [API Endpoints](#api-endpoints)
  - [Set User Details](#set-user-details)
  - [Fetch and Store Transactions](#fetch-and-store-transactions)
  - [Analyze and Detect Anomalies](#analyze-and-detect-anomalies)
  - [Visualization](#visualization)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Troubleshooting](#troubleshooting)
- [Support Information](#support-information)

---

## Overview
This project is an API-based solution designed to detect anomalies in bank account transactions using machine learning. It streamlines the entire process—from transaction retrieval and database storage to anomaly detection via DBSCAN clustering and visualization through PCA dimensionality reduction.

## Features
- Secure storage and management of bank credentials
- Automated transaction retrieval from supported banks
- Efficient transaction data handling and storage
- Robust anomaly detection using DBSCAN clustering
- Interactive visualization of transaction anomalies using PCA

## Prerequisites

### Python Dependencies
Ensure Python 3.8+ is installed along with dependencies listed in `requirements.txt`:

```sh
fastapi==0.115.6
uvicorn==0.34.0
sqlalchemy
numpy
openai
scikit-learn
geopy
forex-python
pandas
matplotlib
pytest-cov
dotenv
```

### Node.js Dependencies
Ensure Node.js v14+ is installed. Dependencies are managed via `package.json`:

```json
{
  "dependencies": {
    "israeli-bank-scrapers": "^5.2.3",
    "dotenv": "^16.0.3"
  }
}
```

### Database Requirements
- PostgreSQL or compatible relational database

## Setup and Installation

### Clone the Repository
```sh
git clone <repository_url>
cd <project_directory>
```

### Install Python Dependencies
```sh
pip install -r requirements.txt
```

### Install Node.js Dependencies
```sh
npm install
```

### Database Setup
Initialize your database:
```sh
python -c "from database import Base, engine; Base.metadata.create_all(bind=engine)"
```

### Start the Server
```sh
uvicorn app:app --reload
```

### Windows-Specific Instructions
- Use Command Prompt or PowerShell
- Verify PostgreSQL installation and ensure `psql` is added to your system PATH

### Mac-Specific Instructions
- Use Terminal
- Install PostgreSQL via Homebrew:
```sh
brew install postgresql
```

## Usage Instructions
Follow these steps to use the application effectively:
1. Set your bank credentials securely.
2. Define parameters for transaction retrieval.
3. Automatically download transactions.
4. Load transactions into the database.
5. Analyze transactions to detect anomalies.
6. Visualize detected anomalies through provided graphical tools.

## API Endpoints

### Set User Details
- **Set Credentials:** `POST /set-credentials`

  Example payload:
  ```json
  {"username": "your_bank_username", "password": "your_bank_password"}
  ```

- **Set Transaction Data:** `POST /set-transData`

  Example payload:
  ```json
  {"company_id": "COMPANY_ID", "start_date": "YYYY-MM-DD"}
  ```

### Fetch and Store Transactions
- **Download Transactions:** `POST /run-transaction`
- **Load Transactions into Database:** `POST /load-transactions`

### Analyze and Detect Anomalies
- **Analyze Transactions:** `GET /analyze-transactions/{user_id}`
  - Optional parameters: `grid_search`, `eps`, `min_samples`

### Visualization
- **Visualize Anomalies:** `GET /visualize-transactions/{user_id}`

## How It Works
1. Securely store your credentials and transaction parameters.
2. Automatically download transactions using Node.js scripts.
3. Store and manage data in PostgreSQL.
4. Detect anomalies using DBSCAN clustering.
5. Visualize anomalies using PCA for dimensionality reduction.

## Technologies Used
- **Backend:** FastAPI, SQLAlchemy, PostgreSQL
- **Machine Learning:** Scikit-learn (DBSCAN, PCA)
- **Scripting:** Node.js

## Troubleshooting
- **Database Issues:** Ensure PostgreSQL is running and accessible.
- **Dependency Errors:** Confirm all dependencies are correctly installed.
- **Node.js Issues:** Verify proper configuration of `index.js`.

## Support Information
For assistance or inquiries, please contact Idan Agami or Tiltan Yaniv:

📧 [idan02486@gmail.com](mailto:idan02486@gmail.com)
📧 [Tiltanyaniv@gmail.com](mailto:Tiltanyaniv@gmail.com)


You can also create an issue directly in this repository.

---