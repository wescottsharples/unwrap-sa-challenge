"""OpenAI prompts constructed from template definitions."""


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from .templates import few_shot_template, zero_shot_template, zero_shot_template_with_title
from .utils import llm

zero_shot_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=zero_shot_template,
        input_variables=["text"],
    ),
)

zero_shot_chain_with_title = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=zero_shot_template_with_title,
        input_variables=["title", "text"],
    ),
)

few_shot_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=few_shot_template,
        input_variables=["title", "text"],
    ),
)
