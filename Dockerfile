# Dockerfile for deploying the Flask DataLab app
FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements early for cache
COPY requirements.txt /main/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /main/requirements.txt

# Copy project
COPY . /main

EXPOSE 8000

# Run the Flask app using gunicorn (assumes `app.py` defines `app`).
# Bind to the port provided by the environment (Vercel sets $PORT for containers).
ENTRYPOINT ["sh", "-c", "gunicorn main:main --bind 0.0.0.0:${PORT:-8000} --workers 2"]
