# Use a Python base image with a lean OS
FROM python:3.9-slim

# Install all necessary system dependencies for tesseract and opencv
RUN apt-get update && apt-get install -y \
    libtesseract-dev \
    libleptonica-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code and the model file
COPY . .

# Set the entry point to run your FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]