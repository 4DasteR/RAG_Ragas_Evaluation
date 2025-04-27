from dotenv import load_dotenv

from components.evaluator import Evaluator
from components.models_provider import LLMFactory, provide_openai_embeddings
from components.query_builder import QueryBuilder
from components.rag_chain import *

if __name__ == "__main__":
    load_dotenv()
    
    llm = LLMFactory.openai()
    embedding_engine = provide_openai_embeddings()
    vectorstore_provider = VectorStoreProvider(embedding_engine)
    
    rags = {
        "simple": RAGFactory.create_simple(llm, vectorstore_provider),
        "compression_rag": RAGFactory.create_compression(llm, vectorstore_provider),
        "hybrid_rag": RAGFactory.create_hybrid(llm, vectorstore_provider)
    }

    # Zero-Shot, Chain of Thought, Role prompting
    prompt_techniques = {
        "zero_shot": "Explain in at least 1 paragraph and at max 3 paragraphs",
        "chain_of_thought": "Please think about this step by step.",
        "role_prompting": "You are an expert in "
    }

    test_queries = [
        "What is a perceptron?",
        "What are the possible ways to implement a composition in java?",
        "How do you implement an overlapping inheritance in java?"
    ]

    engineered_queries: List[Query] = []
    
    roles = ['machine learning basis and algorithmic matters',
             'java programming and know about different association types',
             'java programming and know about different inheritance types']
    
    ground_truth_answers = [
        "Perceptron is the mathematical model of a neuron which outputs a true or false result via a dot product of weight vector and input vector by deciding whether it crosses the threshold or not.",
        "You can implement composition either via normal association with private setter on part and also proper rules od adding method of the whole, or by utilizing inner class wtih private constructor and static method for creation of part.",
        "You can implement overlapping inheritance in java either via flattening of the hierarchy or composition of cardinality one-to-zero between all subclasses."
    ]
    
    for i, query in enumerate(test_queries):
        engineered_queries.append((
            QueryBuilder()
            .text(query)
            .ground_truth_answer(ground_truth_answers[i])
            .zero_shot()
            .chain_of_thought()
            .role_prompting(roles[i])
            # .self_consistency()
            # .directional_stimulus(["Check your answer carefully.", "Check provided context."])
            .build()         
        ))
        
    metrics = ['context_recall', 'faithfulness', 'factual_correctness(mode=f1)']

    discriminator = LLMFactory.openai(temperature=0.3)
    test_q = QueryBuilder().text("What is a perceptron?").ground_truth_answer(ground_truth_answers[0]).zero_shot().chain_of_thought().build()
    evaluator = Evaluator(rags['simple'], discriminator, [test_q])
    res = evaluator.evaluate()
    print(res)