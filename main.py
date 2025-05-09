from typing import List

from dotenv import load_dotenv

from components.evaluator import Evaluator
from components.models_provider import LLMFactory, EmbeddingFactory
from components.query_builder import QueryBuilder, Query
from components.rag_chain import RAGFactory
from components.vector_store import VectorStoreProvider

if __name__ == "__main__":
    load_dotenv()
    
    llm = LLMFactory.openai()
    embedding_engine = EmbeddingFactory.openai()
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

    rags_eval = dict()
    discriminator = LLMFactory.openai(temperature=0.3)
    for rag, instance in rags.items():
        evaluator = Evaluator(instance, discriminator, engineered_queries)
        rags_eval[f"{rag}"] = evaluator.evaluate()
    import pandas as pd
    metrics = ['context_recall', 'faithfulness', 'factual_correctness(mode=f1)']
    for rag_name, eval_res in rags_eval.items():
        print(f"\nSummary for: {rag_name}")

        # Create a dictionary to hold per-metric, per-question mean values
        summary = {}

        # Iterate over questions
        for q_n, df in eval_res.items():
            means = df[metrics].mean()  # Mean of each metric in the DataFrame
            for metric, val in means.items():
                summary.setdefault(metric, {})[q_n] = val  # Fill metric row with question value

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary).T  # Metrics as rows
        summary_df.index.name = "Metric"
        summary_df = summary_df.reset_index()

        print(summary_df)