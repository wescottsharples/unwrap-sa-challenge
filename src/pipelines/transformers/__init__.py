"""Module for retrieving HuggingFace transformer pipelines."""


from transformers import TextClassificationPipeline, pipeline

from src.config import DEVICE, LABEL_MAPPING


def get_transformer_pipelines() -> dict:
    """Get the pre-selected HuggingFace sentiment analysis pipelines.

    Returns:
        dict: dictionary of pipelines
    """

    pipelines = {}

    models = [
        ("roberta_cardiffnlp", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
        ("bert_seethal", "Seethal/sentiment_analysis_generic_dataset"),
        ("roberta_hartmann", "j-hartmann/sentiment-roberta-large-english-3-classes"),
    ]  # NOTE: See `README.md` for details on the models

    for name, model in models:
        pipe = pipeline(
            "sentiment-analysis",
            model=model,
            device=DEVICE,
            max_length=512,
            truncation=True,  # to avoid errors with long texts
        )
        pipe.model.config.id2label = (
            LABEL_MAPPING  # get the pipeline to output strings instead of ints
        )
        pipelines[name] = pipe

    return pipelines
