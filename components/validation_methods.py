from typing import Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM


def validate_string(string: str):
    return isinstance(string, str) and len(string.strip()) > 0

def validate_llm(llm: Union[LLM, BaseChatModel]):
    return isinstance(llm, (LLM, BaseChatModel))