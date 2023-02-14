"""OpenAI pipelines."""


from .pipelines import FewShotPipeline, ZeroShotPipeline, ZeroShotPipelineWithTitle


def get_openai_pipelines() -> dict:
    """Gets the OpenAI pipelines

    Returns:
        dict: The OpenAI pipelines
    """
    print("Loading OpenAI pipelines...")
    return {
        "zero_shot": ZeroShotPipeline(),
        # "zero_shot_with_title": ZeroShotPipelineWithTitle(), # TODO: Test this (didn't have time)
        "few_shot": FewShotPipeline(),
    }
