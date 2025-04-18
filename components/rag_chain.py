from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from components.vector_store import load_and_split_documents
from typing import List, Optional
from dataclasses import dataclass
"""Przerobić to na factory design patter i produkować odpowiednie ragi z fabryki"""
DEFAULT_RETRIEVER_K = 4

DEFAULT_PROMPT_TEMPLATE = """
            You are an AI assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.

            Question: {question}

            Context: {context}

            Answer:
            """


class SimpleRAG:
    def __init__(self, vectorstore: FAISS | None, model: ChatOpenAI, prompt_template: str = DEFAULT_PROMPT_TEMPLATE, k=DEFAULT_RETRIEVER_K):
        self.vectorstore = vectorstore
        self.llm = model
        self.k = k
        self.retriever = self.build_retriever()
        self.prompt_template = prompt_template

        rag_prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": rag_prompt},
            return_source_documents=True
        )

    def build_retriever(self):
        if not self.vectorstore:
            raise ValueError("No vectorstore provided for base retriever.")
        return self.vectorstore.as_retriever(search_kwargs={"k": self.k})

    def query(self, query: str):
        if len(query.strip()) == 0 or type(query) is not str:
            raise ValueError("Query must be provided as a nonempty string.")
        """Query the RAG system with a question"""
        result = self.chain.invoke(query)

        return {
            "query": query,
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }


class CompressionRAG(SimpleRAG):
    def build_retriever(self):
        base_retriever = super().build_retriever()
        compressor = LLMChainExtractor.from_llm(self.llm)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)


class HybridRAG(SimpleRAG):
    def __init__(self, dense_vectorstore: FAISS, model: ChatOpenAI, documents_paths: List[str], k=DEFAULT_RETRIEVER_K):
        self.dense_vectorstore = dense_vectorstore
        self.documents = load_and_split_documents(documents_paths)
        super().__init__(vectorstore=None, model=model, k=k)

    def build_retriever(self):
        dense_retriever = self.dense_vectorstore.as_retriever(search_kwargs={"k": self.k})
        sparse_retriever = BM25Retriever.from_documents(self.documents)
        sparse_retriever.k = self.k

        return EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5]
        )
