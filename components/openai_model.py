import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Tuple
from .logger import Logger

BASE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"
logger = Logger()

def provide(base_model: str = BASE_MODEL, embedding_model: str = EMBEDDING_MODEL, base_temperature=0) -> Tuple[ChatOpenAI, OpenAIEmbeddings]:
    openai_api_key = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(api_key=openai_api_key, model=base_model, temperature=base_temperature)
    embedding_engine = OpenAIEmbeddings(api_key=openai_api_key, model=embedding_model)
    logger.log("OpenAI models provided")

    return llm, embedding_engine