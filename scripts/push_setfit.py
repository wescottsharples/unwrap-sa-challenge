"""Push trained SetFit model to the HuggingFace Hub."""

import os
import pickle

from src.config import SETFIT_MODEL_PATH


def push_model_to_hub():
    """Push trained SetFit model to the HuggingFace Hub."""

    if not os.path.exists(SETFIT_MODEL_PATH):
        raise FileNotFoundError("Model not found. Please train the model first.")

    model = pickle.load(open(SETFIT_MODEL_PATH, "rb"))

    # Push the model to the HuggingFace Hub
    # change model_path to str
    model.push_to_hub("wescottsharples/setfit-23-02-12")


if __name__ == "__main__":
    push_model_to_hub()
