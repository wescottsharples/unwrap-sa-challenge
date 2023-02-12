"""Dataset loading and preprocessing

The dataset should be placed in the `data/raw` folder as a csv file
and the path should be specified in the `config.py` file using the
`DATASET_PATH` variable.
"""


import pandas as pd
from datasets.arrow_dataset import Dataset
from sklearn.model_selection import train_test_split

from src.config import DATASET_PATH, LABEL_MAPPING, RANDOM_SEED


def load_df() -> pd.DataFrame:
    """Load the dataset as a pandas DataFrame

    Returns:
        pd.DataFrame: Dataset as a pandas DataFrame
    """
    df = pd.read_csv(DATASET_PATH)
    INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
    df["label"] = df["sentiment_output"].map(INVERSE_LABEL_MAPPING)
    return df


def split_dataset(df: pd.DataFrame) -> tuple:
    """Split the dataset into train and test sets

    Args:
        df (pd.DataFrame): Dataset as a pandas DataFrame

    Returns:
        tuple: Train and test datasets as pandas DataFrames
    """
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    return train, test


def load_dataset() -> dict:
    """Load the dataset as a HuggingFace dataset

    Returns:
        dict: HuggingFace dataset
    """
    df = load_df()
    train, test = split_dataset(df)
    train_ds = Dataset.from_pandas(train)
    test_ds = Dataset.from_pandas(test)
    return {"train": train_ds, "test": test_ds}
