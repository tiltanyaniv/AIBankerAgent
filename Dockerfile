FROM python:3.11

# Install Node.js and npm (using apt-get; adjust as needed)
RUN apt-get update && apt-get install -y nodejs npm

# Set the working directory in the container
WORKDIR /code

# Copy package.json and package-lock.json for Node dependencies
COPY package*.json ./

# Install Node.js dependencies
RUN npm install

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 3100

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3100", "--workers", "4"]