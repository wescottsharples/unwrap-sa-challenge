"""Functions to get HuggingFace transformer pipelines"""

from transformers import pipeline

from src.config import device


def get_pipes() -> dict:
    """Get the pre-selected HuggingFace pipelines

    Returns:
        dict: dictionary of pipelines
    """

    pipes = {}

    # NOTE: We're doing multi-class sentiment analysis, so we have to keep that in mind when selecting models
    # Our models of choice are:

    # https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    # Fine-tuned version of RoBERTa-base model trained on 123.86M tweets
    CARDIFF_NLP_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    add_pipe("cardiffnlp", CARDIFF_NLP_MODEL, pipes)

    # https://huggingface.co/Seethal/sentiment_analysis_generic_dataset
    # BERT base model (uncased) fine-tuned on a "classified" sentiment analysis dataset
    SEETHAL_MODEL = "Seethal/sentiment_analysis_generic_dataset"
    add_pipe("seethal", SEETHAL_MODEL, pipes)

    # https://huggingface.co/j-hartmann/sentiment-roberta-large-english-3-classes
    # RoBERTa-based model fine-tuned on 5,304 manually annotated social media posts
    HARTMANN_MODEL = "j-hartmann/sentiment-roberta-large-english-3-classes"
    add_pipe("hartmann", HARTMANN_MODEL, pipes)

    return pipes


def add_pipe(name: str, model: str, pipes: dict):
    """Add a pipeline to the dictionary of pipelines

    Args:
        name (str): name of the pipeline
        model (str): HuggingFace model name
        pipes (dict): dictionary of pipelines
    """

    LABEL_MAPPING = {
        0: "NEGATIVE",
        1: "NEUTRAL",
        2: "POSITIVE",
    }  # all models share this mapping

    pipes[name] = pipeline(
        "sentiment-analysis",
        model=model,
        device=device,
        max_length=512,
        truncation=True,  # to avoid errors with long texts
    )
    pipes[name].model.config.id2label = LABEL_MAPPING
