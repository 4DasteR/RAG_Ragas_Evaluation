from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents.base import Document

# Set up default configurations
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def load_and_split_document(file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    """Load a document and split it into chunks"""
    # Determine loader based on file extension
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    # Load the document
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages/segments")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    return chunks


def load_and_split_documents(file_paths: List[str], chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    return sum([load_and_split_document(document, chunk_size, chunk_overlap) for document in file_paths], [])


def create(documents: List[str], embedding_model: OpenAIEmbeddings) -> FAISS:
    """Create a vector store from document chunks"""
    chunks = load_and_split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    print(f"Created vector store with {len(chunks)} documents")
    return vectorstore
