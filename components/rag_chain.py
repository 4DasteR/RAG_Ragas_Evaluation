from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI
from typing import Optional, Union, List, Callable, Dict
from .vector_store import VectorStoreProvider
from .logger import Logger
from dataclasses import dataclass, field
from pydantic import BaseModel
from abc import ABC

DEFAULT_PROMPT_TEMPLATE = """You are an AI assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

QUESTION: {question}

CONTEXT: {context}

Answer:"""

        
logger = Logger()
            
class RetrieverProxy(BaseRetriever, BaseModel):
    """This class in a proxy for retriever to allow dynamic access to it"""
    retriever_callable: Callable[[], BaseRetriever]
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        retriever = self.retriever_callable()
        return retriever.invoke(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        retriever = self.retriever_callable()
        return await retriever.ainvoke(query)

@dataclass
class RAG(ABC):
    llm: ChatOpenAI
    retriever: BaseRetriever
    prompt_template: str
    __chain: Optional[RetrievalQA] = field(default=None, init=False)
    __last_retriever_hash: int = field(default=-1, init=False)
    
    def __post_init__(self):
        logger.log(f"Created {type(self).__name__}.")
        self.__rag_prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    def _get_retriever_hash(self) -> int:
        return hash(str(self.retriever))
    
    @property
    def chain(self) -> RetrievalQA:
        current_hash = self._get_retriever_hash()
        if self.__chain is None or current_hash != self.__last_retriever_hash:
            logger.log(f"Creating RAG chain for {type(self).__name__}...")
            self.__chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": self.__rag_prompt},
                return_source_documents=True
            )
            self.__last_retriever_hash = current_hash
            logger.log("RAG chain created.")
        return self.__chain
    
    def query(self, query: str) -> Dict[str, Union[str, List[Document]]]:
        if len(query.strip()) == 0 or type(query) is not str:
            raise ValueError("Query must be provided as a nonempty string.")
        
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
    
    def _get_retriever_hash(self) -> int:
        if hasattr(self.retriever, 'base_retriever'):
            return hash(str(self.retriever.base_retriever))
        return super()._get_retriever_hash()

@dataclass
class HybridRAG(RAG):
    pass

class RAGFactory(ABC):
    @staticmethod
    def __validate_inputs(llm: ChatOpenAI, provider: VectorStoreProvider, prompt_template: str):
        if not llm:
            raise ValueError("LLM must be provided!")
        if not provider:
            raise ValueError("Provider must be provided!")
        if not prompt_template:
            raise ValueError("Prompt template must be provided!")
    
    @staticmethod
    def create_simple(
        llm: ChatOpenAI,
        provider: VectorStoreProvider,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    ) -> RAG:
        RAGFactory.__validate_inputs(llm, provider, prompt_template)
        retriever_callable = lambda: provider.dense_retriever
        return SimpleRAG(llm, RetrieverProxy(retriever_callable=retriever_callable), prompt_template)
    
    @staticmethod
    def create_compression(
        llm: ChatOpenAI,
        provider: VectorStoreProvider,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    ) -> RAG:
        RAGFactory.__validate_inputs(llm, provider, prompt_template)
        retriever_callable = lambda: provider.dense_retriever
        return CompressionRAG(llm, RetrieverProxy(retriever_callable=retriever_callable), prompt_template)
    
    @staticmethod
    def create_hybrid(
        llm: ChatOpenAI,
        provider: VectorStoreProvider,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    ) -> RAG:
        RAGFactory.__validate_inputs(llm, provider, prompt_template)
        retriever_callable = lambda: provider.ensemble_retriever
        return HybridRAG(llm, RetrieverProxy(retriever_callable=retriever_callable), prompt_template)