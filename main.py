from components import openai_model
from components.vector_store import VectorStoreProvider
from components.rag_chain import *
from components.evaluator import Evaluator
import pandas as pd
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    # documents = [f"documents/{f}" for f in os.listdir('documents')]

    """Start Rag pipeline"""
    llm, embedding_engine = openai_model.provide()
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

    engineered_queries: list[dict[str, str]] = []
    
    roles = ['machine learning basis and algorithmic matters.',
             'java programming and know about different association types.',
             'java programming and know about different inheritance types.']
    
    for i, query in enumerate(test_queries):
        engineered_queries.append({
            "zero_shot": query + "\n" + prompt_techniques["zero_shot"],
            "chain_of_thought": query + "\n" + prompt_techniques["chain_of_thought"],
            "role_prompting": prompt_techniques["role_prompting"]
                              + f"{roles[i]}"
                              + "\n" + query
        })

    ground_truth_answers = [
        "Perceptron is the mathematical model of a neuron which outputs a true or false result via a dot product of weight vector and input vector by deciding whether it crosses the threshold or not.",
        "You can implement composition either via normal association with private setter on part and also proper rules od adding method of the whole, or by utilizing inner class wtih private constructor and static method for creation of part.",
        "You can implement overlapping inheritance in java either via flattening of the hierarchy or composition of cardinality one-to-zero between all subclasses."
    ]

    discriminator = openai_model.provide(base_temperature=0.3)[0]
    rags_eval = dict()
    for (rag_name, rag_instance) in rags.items():
        evaluator = Evaluator(rag_instance, discriminator, engineered_queries, ground_truth_answers)
        rags_eval[rag_name] = evaluator.evaluate()
        evaluator.save_to_json()
        
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