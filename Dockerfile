# Dockerfile for deploying the Flask DataLab app
FROM python:3.10.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps required for scientific packages (scipy, numpy wheels), xgboost, pycaret, etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       g++ \
       cmake \
       git \
       libatlas-base-dev \
       libopenblas-dev \
       liblapack-dev \
       gfortran \
       libssl-dev \
       libffi-dev \
       pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements early to leverage Docker layer cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install wheel/setuptools first, then install requirements
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY . /app

# Expose default development port; container will bind to $PORT when deployed
EXPOSE 8000

# Run the Flask app using gunicorn; bind to the PORT environment variable when provided
CMD ["python", "Final_data/DataLab/main.py"]
