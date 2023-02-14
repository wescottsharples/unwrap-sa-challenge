"""SetFit pipeline"""


from typing import Callable

from src.config import LABEL_MAPPING

from .loading import load_setfit_model


def get_setfit_pipeline() -> dict[str, Callable]:
    """Get the SetFit pipeline

    Args:
        model (SetFitModel): SetFit model

    Returns:
        Callable: SetFit pipeline
    """
    try:
        model = load_setfit_model()
    except FileNotFoundError:
        raise FileNotFoundError(
            "SetFit model not found. Please run `python scripts/train_setfit.py` to train the model."
        )

    def setfit_pipeline(texts: list[str]) -> list[str]:
        """SetFit pipeline as a callable function

        Args:
            texts (list[str]): list of texts

        Returns:
            list[str]: list of labels
        """
        preds = model.predict(texts)
        # preds is a tensor of shape (len(texts), 1)
        return [LABEL_MAPPING[pred.item()] for pred in preds]  # type: ignore

    return {"setfit": setfit_pipeline}
