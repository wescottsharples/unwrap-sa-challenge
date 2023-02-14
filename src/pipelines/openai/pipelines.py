"""Module defines callable OpenAI pipelines."""

import asyncio

import tiktoken

from src.config import VALID_SENTIMENTS

from .prompts import few_shot_chain, zero_shot_chain, zero_shot_chain_with_title
from .utils import get_best_match, llm


def parse_completion(completion: str, fuzzy: bool = True) -> str:
    """Parses the sentiment from the pipeline completion.

    Example:
        >>> parse_zero_shot_completion(" positive")
        "POSITIVE"
        >>> parse_zero_shot_completion(" neg")
        "NEGATIVE"

    Args:
        completion (str): The completion from the zero-shot pipeline
        fuzzy (bool, optional): Whether to use fuzzy matching. Defaults to True.

    Returns:
        str: The predicted sentiment (NEGATIVE, NEUTRAL, POSITIVE)
    """
    parsed = completion.strip().upper()
    if parsed not in VALID_SENTIMENTS:
        if fuzzy:
            parsed = get_best_match(parsed)
        else:
            raise ValueError(f"Unexpected sentiment: {parsed}")
    return parsed


class OpenAIPipeline:
    """Base class for OpenAI pipelines."""

    max_text_tokens = 400
    is_async = True

    async def __call__(self, *args, **kwargs) -> str:
        raise NotImplementedError

    def truncate_text(self, text: str) -> str:
        """Truncates the text to the maximum number of tokens allowed by the pipeline.

        Args:
            text (str): The text to tokenize

        Returns:
            text (str): The truncated text
        """
        encoder = "p50k_base"
        # create a GPT-3 encoder instance
        enc = tiktoken.get_encoding(encoder)

        # encode the text using the GPT-3 encoder
        tokenized_text = enc.encode(text)

        # truncate the text if it is too long
        if len(tokenized_text) > self.max_text_tokens:
            tokenized_text = tokenized_text[: self.max_text_tokens]

        # decode the tokenized text
        return enc.decode(tokenized_text)


class ZeroShotPipeline(OpenAIPipeline):
    """Zero-shot sentiment analysis pipeline."""

    expects_titles = False

    async def __call__(self, texts: list[str]) -> list[str]:
        texts = [self.truncate_text(text) for text in texts]  # preprocess
        tasks = [zero_shot_chain.arun(text=text) for text in texts]
        completions = await asyncio.gather(*tasks)  # run in parallel
        return [parse_completion(completion) for completion in completions]  # postprocess


class ZeroShotPipelineWithTitle(OpenAIPipeline):
    """Zero-shot sentiment analysis pipeline with title support."""

    expects_titles = True

    async def __call__(self, title_text_pairs: list[tuple[str, str]]) -> list[str]:
        title_text_pairs = [(title, self.truncate_text(text)) for title, text in title_text_pairs]
        tasks = [
            zero_shot_chain_with_title.arun(title=title, text=text)
            for title, text in title_text_pairs
        ]
        completions = await asyncio.gather(*tasks)
        return [parse_completion(completion) for completion in completions]


class FewShotPipeline(OpenAIPipeline):
    """Few-shot sentiment analysis pipeline with title support."""

    expects_titles = True

    async def __call__(self, title_text_pairs: list[tuple[str, str]]) -> list[str]:
        title_text_pairs = [(title, self.truncate_text(text)) for title, text in title_text_pairs]
        tasks = [few_shot_chain.arun(title=title, text=text) for title, text in title_text_pairs]
        completions = await asyncio.gather(*tasks)
        return [parse_completion(completion) for completion in completions]
