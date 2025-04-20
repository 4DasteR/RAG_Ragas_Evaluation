from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import List, Optional, Set, Dict, Type
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import VectorStoreRetriever
from dataclasses import dataclass, field
from langchain.retrievers import EnsembleRetriever
from pathlib import Path
from itertools import chain
from .logger import Logger

logger = Logger()

@dataclass
class VectorStoreProvider:
    embedding_model: OpenAIEmbeddings
    k: int = 4
    chunk_size: int = 1000
    chunk_overlap: int = 200
    weight_dense: float = 0.5
    weight_sparse: float = 0.5
    documents_path: Path = Path("documents")
    __dense_retriever: Optional[VectorStoreRetriever] = field(default=None, init=False)
    __sparse_retriever: Optional[BM25Retriever] = field(default=None, init=False)
    __cached_documents: Dict[Path, float] = field(default_factory=dict, init=False)
    __loaders: Dict[str, Type] = field(init=False, default_factory=lambda: {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader
    })
    
    def __post_init__(self):
        logger.log("Vector store provider created.")
        self.documents_path.mkdir(exist_ok=True)
        self.__documents_changed_check()
    
    @property
    def __document_paths(self) -> Set[Path]:
        return set(self.documents_path.glob("*.*"))
        
    def __documents_changed_check(self) -> bool:
        current = {path: path.stat().st_mtime for path in self.__document_paths}
        if current != self.__cached_documents:
            logger.log("Documents changed. Updating cache...")
            self.__cached_documents = current
            self.__dense_retriever = None
            self.__sparse_retriever = None
            return True
        return False
            
    def __validate_document(self, path: Path) -> bool:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    
    def __load_and_split_document(self, path: Path) -> List[Document]:
        if not self.__validate_document(path):
                raise FileNotFoundError(f"Invalid document file: {path}")

        suffix = path.suffix
        loader = self.__loaders.get(path.suffix)
        if not loader:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        documents = loader(str(path)).load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        return text_splitter.split_documents(documents)

    def __load_and_split_documents(self) -> List[Document]:
        if not hasattr(self, '__split_documents') or self.__documents_changed_check():
                self.__split_documents = list(chain.from_iterable(self.__load_and_split_document(doc) for doc in self.__cached_documents))
        return self.__split_documents
    
    @property
    def dense_retriever(self) -> VectorStoreRetriever:
        if not self.__dense_retriever or self.__documents_changed_check():
            logger.log("Building dense retriever...")
            chunks = self.__load_and_split_documents()
            vectorstore = FAISS.from_documents(chunks, self.embedding_model)
            self.__dense_retriever = vectorstore.as_retriever(search_kwargs={"k": self.k})
            logger.log("Build complete.")
        return self.__dense_retriever
    
    @property
    def sparse_retriever(self) -> BM25Retriever:
        if not self.__sparse_retriever or self.__documents_changed_check():
            logger.log("Building sparse retriever...")
            chunks = self.__load_and_split_documents()
            self.__sparse_retriever = BM25Retriever.from_documents(chunks)
            logger.log("Build complete.")
        return self.__sparse_retriever
    
    @property
    def ensemble_retriever(self) -> EnsembleRetriever:
        return EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[self.weight_dense, self.weight_sparse]
        )