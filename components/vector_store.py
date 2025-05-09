from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import List, Optional, Set, Dict, Type

from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader, UnstructuredMarkdownLoader, \
    UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from .logger import Logger
from .validation_methods import validate_string

logger = Logger()


@dataclass
class VectorStoreProvider:
    """
    Class responsible for providing dense and sparse retrievers for vectorstore. It automatically detects changes to documents and rebuilds itself.
    Supports only TXT and PDF files for documents.

    Attributes:
        embedding_model (Embeddings): model used for text tokenization and embedding
        chunk_size (int): size of document chunk. Default: 1000
        chunk_overlap (int): number of chunk overlaps. Default: 200
        weight_dense (float): The percentage of how much dense retriever is allocated in Ensemble retriever. Default: 0.5
        weight_sparse (float): The percentage of how much sparse retriever is allocated in Ensemble retriever. Default: 0.5
        documents_path (Path): Path to documents directory. Default: Path("documents").
    """
    embedding_model: Embeddings
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
        ".txt": TextLoader,
        ".html": BSHTMLLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".pptx": UnstructuredPowerPointLoader,
    })

    def __post_init__(self):
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError("k must be over 0!")

        if not isinstance(self.weight_dense, float) or (0 >= self.weight_dense or self.weight_dense >= 1):
            raise ValueError("Dense weight must be in range (0, 1)")

        if not isinstance(self.weight_sparse, float) or (0 >= self.weight_sparse or self.weight_sparse >= 1):
            raise ValueError("Sparse weight must be in range (0, 1)")

        if self.weight_dense + self.weight_sparse != 1:
            raise ValueError("Both weights must sum up to 1!")

        if not isinstance(self.chunk_size, int) or self.chunk_size <= 200:
            raise ValueError("Chunk size must be at least 200!")

        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap mustn't be bigger or equal the size of chunk!")

        if not isinstance(self.documents_path, Path):
            if not validate_string(self.documents_path):
                raise ValueError("Path to documents must be a Path object!")
            else:
                self.documents_path = Path(self.documents_path)

        logger.log("Vector store provider created.", "COMPLETED")
        self.documents_path.mkdir(exist_ok=True)
        self.__documents_changed_check()

    @property
    def documents_files(self) -> Set[Path]:
        return set(self.documents_path.glob("*.*"))

    def __documents_changed_check(self) -> bool:
        current = {path: path.stat().st_mtime for path in self.documents_files}
        if current != self.__cached_documents:
            logger.warn("Documents changed. Updating cache...", "DOCUMENTS")
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
        if self.__documents_changed_check() or not hasattr(self, '__chunks'):
            if len(self.documents_files) == 0:
                raise ValueError("There must be at least one document in documents folder!")
            self.__chunks = list(chain.from_iterable(self.__load_and_split_document(doc) for doc in self.__cached_documents.keys()))
        return self.__chunks

    @property
    def dense_retriever(self) -> VectorStoreRetriever:
        if not self.__dense_retriever or self.__documents_changed_check():
            logger.log("Building dense retriever...", "CREATION")
            chunks = self.__load_and_split_documents()
            vectorstore = FAISS.from_documents(chunks, self.embedding_model)
            self.__dense_retriever = vectorstore.as_retriever(search_kwargs={"k": self.k})
            logger.log("Build complete.", "COMPLETED")
        return self.__dense_retriever

    @property
    def sparse_retriever(self) -> BM25Retriever:
        if not self.__sparse_retriever or self.__documents_changed_check():
            logger.log("Building sparse retriever...", "CREATION")
            chunks = self.__load_and_split_documents()
            self.__sparse_retriever = BM25Retriever.from_documents(chunks)
            logger.log("Build complete.", "COMPLETED")
        return self.__sparse_retriever

    @property
    def ensemble_retriever(self) -> EnsembleRetriever:
        return EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[self.weight_dense, self.weight_sparse]
        )
