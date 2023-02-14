"""Module for retrieving all sentiment analysis pipelines under consideration.

A pipeline is a callable that takes a list of texts and returns a list of sentiment labels.
"""

from typing import Callable

from .openai import get_openai_pipelines
from .setfit import get_setfit_pipeline
from .transformers import get_transformer_pipelines


def get_all_pipelines() -> dict[str, Callable]:
    """Get all pipelines for sentiment analysis

    Returns:
        dict: Nested dictionary of pipelines with the following structure:
            {
                "hf": {
                    "bert": bert_pipeline,
                    "roberta": roberta_pipeline,
                    ...
                },
                "openai": {
                    "zero_shot": zero_shot_pipeline,
                    "few_shot": few_shot_pipeline,
                    ...
                },
                "setfit": {
                    "setfit": setfit_pipeline,
                }
    """
    pipelines = {}
    for name, pipelines_generator in [
        ("hf", get_transformer_pipelines),
        ("openai", get_openai_pipelines),
        ("setfit", get_setfit_pipeline),
    ]:
        pipelines[name] = {}
        for pipeline_name, pipeline in pipelines_generator().items():
            pipelines[name][pipeline_name] = pipeline
    return pipelines
