"""Module for loading and preprocessing our sentiment analysis dataset."""


import os

import pandas as pd
from datasets import Dataset, DatasetDict  # type: ignore
from sklearn.model_selection import train_test_split

from src.config import DATASET_PATH, LABEL_MAPPING, RANDOM_SEED
from src.connections import get_mysql_connection


def split_dataset(df: pd.DataFrame) -> tuple:
    """Split the dataset into train and test sets

    Args:
        df (pd.DataFrame): Dataset as a pandas DataFrame

    Returns:
        tuple: Train and test datasets as pandas DataFrames
    """
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    return train, test


def add_label_column(df: pd.DataFrame) -> None:
    """Add a "label" column to the DataFrame with integer values

    Args:
        df (pd.DataFrame): Dataset as a pandas DataFrame
    """
    INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
    label_key = "sentiment_output" if "sentiment_output" in df.columns else "sentiment"
    df["label_str"] = df[label_key]  # alias for the original label
    df["label"] = df[label_key].map(INVERSE_LABEL_MAPPING)


def load_dataset_from_file(dataset_path: os.PathLike) -> dict:
    """Load the dataset as a HuggingFace dataset

    Returns:
        dict: HuggingFace dataset
    """
    df = pd.read_csv(dataset_path)
    add_label_column(df)
    train, test = split_dataset(df)
    return DatasetDict({"train": Dataset.from_pandas(train), "test": Dataset.from_pandas(test)})


def load_dataset_from_mysql() -> dict:
    """Load the dataset as a HuggingFace dataset from MySQL

    Returns:
        dict: HuggingFace dataset
    """
    print("Loading dataset from MySQL...")
    conn = get_mysql_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM feedback_entries")
        result = cursor.fetchall()
    df = pd.DataFrame(result)
    df.columns = [i[0] for i in cursor.description]
    add_label_column(df)
    # No need to split since we consider the whole dataset as the test set
    return DatasetDict({"test": Dataset.from_pandas(df)})


def load_datasets() -> tuple[dict, dict]:
    """Load the latest datasets from both file and MySQL

    Returns:
        tuple[dict, dict]: Datasets as HuggingFace datasets (file, MySQL)
    """
    return load_dataset_from_file(DATASET_PATH), load_dataset_from_mysql()


def load_latest_test_dataset() -> Dataset:
    """Load the latest test data from MySQL

    NOTE: Removes any examples which appear in file's training data

    Returns:
        dict: Test dataset as a HuggingFace dataset
    """
    file_dataset, mysql_dataset = load_datasets()
    file_train_df = file_dataset["train"].to_pandas()
    mysql_test_df = mysql_dataset["test"].to_pandas()
    # Remove any examples which appear in file's training data
    mysql_test_df = mysql_test_df[~mysql_test_df["id"].isin(file_train_df["id"])]
    return Dataset.from_pandas(mysql_test_df)


def load_latest_train_dataset() -> Dataset:
    """Load the latest training data from file

    Returns:
        dict: Training dataset as a HuggingFace dataset
    """
    dataset = load_dataset_from_file(DATASET_PATH)
    return dataset["train"]
