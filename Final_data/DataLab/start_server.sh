#!/bin/bash

# Load environment variables from .env file
set -a
source .env
set +a

# Start the Flask application
python3 main.py
