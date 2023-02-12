import os
from pathlib import Path

import torch

# Defines the configuration for the application
RANDOM_SEED = 42

# MySQL database configuration
HOST = os.environ["MYSQL_HOST"]
PORT = os.environ["MYSQL_PORT"]
USER = os.environ["MYSQL_USER"]
PASSWORD = os.environ["MYSQL_PASSWORD"]
DB = os.environ["MYSQL_DB"]
CONNECTION_STRING = f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

PARENT_DIR = Path(__file__).parent.parent
SENTIMENT_ANNOTATIONS_CSV = PARENT_DIR / "data" / "raw" / "sentiment_annotations.csv"
DATASET_PATH = SENTIMENT_ANNOTATIONS_CSV
