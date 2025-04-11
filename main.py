from components import openai_model, vector_store
from components.rag_chain import SimpleRAG, CompressionRAG, HybridRAG

if __name__ == "__main__":
    documents = ['RAG_data\Ethics-Toolkit.pdf']
    # Start Rag pipeline
    llm, embedding_engine = openai_model.provide()
    vectorstore = vector_store.create(documents, embedding_engine)
    rag = SimpleRAG(vectorstore, llm)
    # rag = CompressionRAG(vectorstore, llm)
    # rag = HybridRAG(vectorstore, llm, documents)

    test_query = "What are tools of ethics? Enumerate and explain briefly all of them."
    result = rag.query(test_query)
    print(f"\nQuery: {result['query']}")
    print(f"Answer: {result['answer']}")
    for i, doc in enumerate(result["source_documents"][:2]):  # Show only first 2 sources for brevity
        print(f"Document {i + 1}:")
        print(doc.page_content[:150], "...")