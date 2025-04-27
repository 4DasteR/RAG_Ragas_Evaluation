from dataclasses import dataclass, field
from typing import Literal, Optional
from abc import ABC
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms.koboldai import KoboldApiLLM
from .validation_methods import *
from .logger import Logger
import os
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

logger = Logger()

class LLMFactory(ABC):
    @staticmethod
    def __validateTemperature(temperature: float):
        return isinstance(temperature, (float, int)) and (temperature >= 0 and temperature <= 1)
    
    @staticmethod
    def openai(model: Optional[str] = "gpt-4o-mini", temperature: float = 0) -> ChatOpenAI:
        """Provides an OpenAI LLM model according to given parameters.

        Args:
            model (str, optional): model for LLM. Defaults to "gpt-4o-mini".
            temperature (float, optional): base temperature of the LLM. Defaults to 0.

        Raises:
            ValueError: No OPENAI_API_KEY provided in .env!
            ValueError: Model name must be a nonempty string!
            ValueError: Temperature must be in range [0, 1]!

        Returns:
            ChatOpenAI: OpenAI LLM model
        """
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Please provide OPENAI_API_KEY to the .env file!")
        
        if not validate_string(model):
            raise ValueError("Model name must be a nonempty string!")
        
        if not LLMFactory.__validateTemperature(temperature):
            raise ValueError("Temperature must be in range [0, 1]!")
        
        logger.log("Provided OpenAI LLM model", 'AI_MODEL')
        return ChatOpenAI(api_key=openai_api_key, model=model, temperature=temperature)
    
    @staticmethod
    def koboldAPI(endpoint: Optional[str] = None, temperature: float = 0, max_length: int = 500) -> KoboldApiLLM:
        """Provides a Kobold LLM model from API according to given parameters. The endpoint is loaded by default from .env with KOBOLD_API, if not specified it will be loaded from parameter.

        Args:
            endpoint (str, optional): API for kobold model. Defaults to None.
            temperature (float, optional): base temperature of the LLM. Defaults to 0.
            max_length (int, optional): maximal number of tokens of the response. Defaults to 500.

        Raises:
            ValueError: Endpoint must be provided as a nonempty string!
            ValueError: Temperature must be in range [0, 1]!
            ValueError: Max number of response tokens must be at least 100!

        Returns:
            KoboldApiLLM: Kobold LLM from API
        """
        
        kobold_api = os.getenv("KOBOLD_API")
        if not kobold_api:
            if not validate_string(endpoint):
                raise ValueError("Endpoint must be provided as a nonempty string!")
            else: 
                kobold_api = endpoint
                
        if not LLMFactory.__validateTemperature(temperature):
            raise ValueError("Temperature must be in range [0, 1]!")
        
        if not isinstance(max_length, int) or max_length < 100:
            raise ValueError("Max number of response tokens must be at least 100!")
                
        logger.log(f"Provided Kobold LLM model from API: {kobold_api}", 'AI_MODEL')
        return KoboldApiLLM(endpoint=kobold_api, temperature=temperature, max_length=max_length)
    
def provide_openai_embeddings():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please provide OPENAI_API_KEY to the .env file!")
    
    logger.log("Provided OpenAI embedding model", 'AI_MODEL')
    return OpenAIEmbeddings(api_key=openai_api_key, model=OPENAI_EMBEDDING_MODEL)