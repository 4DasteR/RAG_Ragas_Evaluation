from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

DEFAULT_RETRIEVER_K = 4


class RAG:
    def __init__(self, vectorstore, model, k=DEFAULT_RETRIEVER_K):
        self.vectorstore = vectorstore
        self.llm = model
        # Create a retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        # Custom prompt template for RAG
        rag_prompt_template = """
            You are an AI assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know.

            Question: {question}

            Context: {context}

            Answer:
            """

        RAG_PROMPT = PromptTemplate(
            template=rag_prompt_template,
            input_variables=["context", "question"]
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": RAG_PROMPT},
            return_source_documents=True
        )

    def query(self, query):
        """Query the RAG system with a question"""
        result = self.chain.invoke(query)

        return {
            "query": query,
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }