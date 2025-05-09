from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Union, List, Callable, Dict, Literal

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents.base import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel

from .logger import Logger
from .query_builder import Query
from .validation_methods import validate_string, validate_llm
from .vector_store import VectorStoreProvider

DEFAULT_PROMPT_TEMPLATE = """You are an AI assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

QUESTION: {question}

CONTEXT: {context}

Answer:"""

        
logger = Logger()
            
class RetrieverProxy(BaseRetriever, BaseModel):
    """
    Proxy class for retriever. Used for dynamic access to actual retriever.

    Attributes:
        retriever_callable(Callable[[], BaseRetriever]): lambda providing access to actual retriever
    """
    retriever_callable: Callable[[], BaseRetriever]
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        retriever = self.retriever_callable()
        return retriever.invoke(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        retriever = self.retriever_callable()
        return await retriever.ainvoke(query)

@dataclass
class RAG(ABC):
    """
    Abstract class representing a RAG.

    Attributes:
        llm (LLM | BaseChatModel): llm model used for querying
        retriever (BaseRetriever): retriever for vectorstore data
        prompt_template (str): template of a prompt enriched with query and context from chain
    """
    llm: Union[LLM, BaseChatModel]
    retriever: BaseRetriever
    prompt_template: str
    __chain: Optional[RetrievalQA] = field(default=None, init=False)
    __last_retriever_hash: int = field(default=-1, init=False)
    
    def __post_init__(self):
        if not validate_llm(self.llm):
            raise ValueError("LLM must be of type LLM or BaseChatModel!")
        
        if not isinstance(self.retriever, BaseRetriever):
            raise ValueError("Retriever must be a valid BaseRetriever!")
        
        if not validate_string(self.prompt_template):
            raise ValueError("Prompt template must be a valid string!")
        
        logger.log(f"Created {type(self).__name__}.", "COMPLETED")
        self.__rag_prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    def __get_retriever_hash(self) -> int:
        return hash(str(self.retriever))
    
    @property
    def chain(self) -> RetrievalQA:
        current_hash = self.__get_retriever_hash()
        if self.__chain is None or current_hash != self.__last_retriever_hash:
            logger.log(f"Creating RAG chain for {type(self).__name__}...", "CREATION")
            self.__chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": self.__rag_prompt},
                return_source_documents=True
            )
            self.__last_retriever_hash = current_hash
            logger.log("RAG chain created.", "COMPLETED")
        return self.__chain
    
    def query(self, query: Union[str, Query]) -> Dict[Literal["query", "answer", "source_documents"], Union[str, List[Document]]]:
        if isinstance(query, Query):
            query = query.text
        
        if not validate_string(query):
            raise ValueError("Query must be provided as a nonempty string!")
        
        logger.log(f"Asking the query to {type(self).__name__}", "QUERY")
        result = self.chain.invoke(query)
        
        return {
            "query": query,
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }


@dataclass
class SimpleRAG(RAG):
    pass

@dataclass
class CompressionRAG(RAG):
    def __post_init__(self):
        super().__post_init__()
        compressor = LLMChainExtractor.from_llm(self.llm)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=self.retriever
        )
    
    def __get_retriever_hash(self) -> int:
        if hasattr(self.retriever, 'base_retriever'):
            return hash(str(self.retriever.base_retriever))
        return super().__get_retriever_hash()

@dataclass
class HybridRAG(RAG):
    pass

class RAGFactory(ABC):
    """
    Factory class used for generating Simple, Compression and Hybrid RAGs.
    """
    @staticmethod
    def __validate_inputs(llm: Union[LLM, BaseChatModel], provider: VectorStoreProvider, prompt_template: str):
        if not llm:
            raise ValueError("LLM must be provided!")
        if not provider:
            raise ValueError("Provider must be provided!")
        if not prompt_template:
            raise ValueError("Prompt template must be provided!")
    
    @staticmethod
    def create_simple(
        llm: Union[LLM, BaseChatModel],
        provider: VectorStoreProvider,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    ) -> RAG:
        """Creates a simple RAG

        Args:
            llm (LLM | BaseChatModel): llm model used for querying
            provider (VectorStoreProvider): vectorstore provider for access to dense retriever
            prompt_template (str, optional): prompt template for RAG. Defaults to DEFAULT_PROMPT_TEMPLATE.

        Returns:
            RAG: Simple RAG
        """
        RAGFactory.__validate_inputs(llm, provider, prompt_template)
        retriever_callable = lambda: provider.dense_retriever
        return SimpleRAG(llm, RetrieverProxy(retriever_callable=retriever_callable), prompt_template)
    
    @staticmethod
    def create_compression(
        llm: Union[LLM, BaseChatModel],
        provider: VectorStoreProvider,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    ) -> RAG:
        """Creates a compression RAG. 
        Compression RAG uses ContextualCompressionRetriever on the provided retriever.

        Args:
            llm (LLM | BaseChatModel): llm model used for querying
            provider (VectorStoreProvider): vectorstore provider for access to dense retriever
            prompt_template (str, optional): prompt template for RAG. Defaults to DEFAULT_PROMPT_TEMPLATE.

        Returns:
            RAG: Compression RAG
        """
        RAGFactory.__validate_inputs(llm, provider, prompt_template)
        retriever_callable = lambda: provider.dense_retriever
        return CompressionRAG(llm, RetrieverProxy(retriever_callable=retriever_callable), prompt_template)
    
    @staticmethod
    def create_hybrid(
        llm: Union[LLM, BaseChatModel],
        provider: VectorStoreProvider,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    ) -> RAG:
        """Creates a RAG that uses two retrievers based on their ensembling weights.

        Args:
            llm (LLM | BaseChatModel): llm model used for querying
            provider (VectorStoreProvider): vectorstore provider for access to dense and sparse retrievers
            prompt_template (str, optional): prompt template for RAG. Defaults to DEFAULT_PROMPT_TEMPLATE.

        Returns:
            RAG: Hybrid RAG
        """
        RAGFactory.__validate_inputs(llm, provider, prompt_template)
        retriever_callable = lambda: provider.ensemble_retriever
        return HybridRAG(llm, RetrieverProxy(retriever_callable=retriever_callable), prompt_template)