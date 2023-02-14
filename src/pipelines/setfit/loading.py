import os
import pickle

from setfit import SetFitModel

from src.config import SETFIT_MODEL_PATH


def load_local_setfit_model() -> SetFitModel:
    """Load the SetFit model

    Returns:
        SetFitModel: SetFit model
    """
    return pickle.load(open(SETFIT_MODEL_PATH, "rb"))


def load_setfit_model() -> SetFitModel:
    """Load the SetFit model

    Returns:
        SetFitModel: SetFit model
    """
    # if it exists locally, load it
    print(f"Loading SetFit pipeline...")
    if os.path.exists(SETFIT_MODEL_PATH):
        return load_local_setfit_model()
    # otherwise, download it from the HuggingFace Hub
    else:
        print("Downloading SetFit model from HuggingFace Hub...")
        return SetFitModel.from_pretrained("wescottsharples/setfit-23-02-12")
