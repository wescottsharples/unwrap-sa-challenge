"""Configuration file for the application."""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import logging

load_dotenv()  # Load environment variables from .env file
logging.set_verbosity_error()  # Disable transformers warnings

# Training Parameters
RANDOM_SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
VALID_SENTIMENTS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL_MAPPING = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
OPENAI_MODEL_NAME = "text-davinci-003"  # Most expensive OpenAI model

# MySQL Configuration
HOST = os.environ.get("MYSQL_HOST", None)
PORT = os.environ.get("MYSQL_PORT", None)
USER = os.environ.get("MYSQL_USER", None)
PASSWORD = os.environ.get("MYSQL_PASSWORD", None)
DB = os.environ.get("MYSQL_DB", None)
if all([HOST, PORT, USER, PASSWORD, DB]):
    CONNECTION_STRING = f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"
else:
    CONNECTION_STRING = None
    print("No MySQL database configuration found. Add environment variables to .env file.")

# Paths
PARENT_DIR = Path(__file__).parent.parent
DATA_DIR = PARENT_DIR / "data"
CUSTOM_MODELS_DIR = DATA_DIR / "models"
SENTIMENT_ANNOTATIONS_CSV = DATA_DIR / "raw" / "sentiment_annotations.csv"
DATASET_PATH = SENTIMENT_ANNOTATIONS_CSV
SETFIT_MODEL_PATH = CUSTOM_MODELS_DIR / "setfit_model.pkl"

# Create directories if they don't exist
CUSTOM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
