"""Shared OpenAI resources such as the LLM object and util funcs."""


import langchain
from langchain.cache import SQLiteCache
from langchain.llms import OpenAI
from thefuzz import process

from src.config import OPENAI_MODEL_NAME, VALID_SENTIMENTS

# Enable caching of LLM responses
langchain.llm_cache = SQLiteCache(database_path=".openai.db")


llm = OpenAI(
    model_name=OPENAI_MODEL_NAME,
    max_tokens=2,
    temperature=0,  # deterministic
)  # type: ignore


def get_best_match(completion: str) -> str:
    """Returns the best match for the completion

    Args:
        completion (str): The completion from the zero-shot pipeline

    Returns:
        str: The predicted sentiment (NEGATIVE, NEUTRAL, POSITIVE)
    """
    best_match = process.extractOne(completion, VALID_SENTIMENTS, score_cutoff=20)
    if best_match is None:
        raise ValueError(f"Unexpected sentiment: {completion}")
    return best_match[0]
