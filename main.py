from components import openai_model, vector_store
from components.rag_chain import SimpleRAG, CompressionRAG, HybridRAG
import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
from components.evaluator import create_evaluation_dataset, create_evaluation_dicts, evaluate_datasets
from components.utility import *
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    documents = ['documents\perceptron3-en.pdf', 'documents\MAS-06-en.pdf', 'documents\MAS-07-en.pdf']
    # Start Rag pipeline
    # llm, embedding_engine = openai_model.provide()
    # vectorstore = vector_store.create(documents, embedding_engine)
    #
    # simple_rag = SimpleRAG(vectorstore, llm)
    # compression_rag = CompressionRAG(vectorstore, llm)
    # hybrid_rag = HybridRAG(vectorstore, llm, documents)
    #
    # rags = {
    #     "simple": simple_rag,
    #     "compression_rag": compression_rag,
    #     "hybrid_rag": hybrid_rag
    # }

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

    # Engineer the queries with prompt techniques
    engineered_queries: list[dict] = []
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

    # Reference answers for evaluation
    ground_truth_answers = [
        "Perceptron is the mathematical model of a neuron which outputs a true or false result via a dot product of weight vector and input vector by deciding whether it crosses the threshold or not.",
        "You can implement composition either via normal association with private setter on part and also proper rules od adding method of the whole, or by utilizing inner class wtih private constructor and static method for creation of part.",
        "You can implement overlapping inheritance in java either via flattening of the hierarchy or composition of cardinality one-to-zero between all subclasses."
    ]

    # rags_eval_datasets = dict()
    # for (rag_name, rag_instance) in rags.items():
    #     rags_eval_datasets[rag_name] = create_evaluation_dicts(rag_instance, engineered_queries, ground_truth_answers)
    #
    # with open('rags.json', 'w', encoding='utf-8') as f:
    #     json.dump(rags_eval_datasets, f, indent=2, ensure_ascii=False)

    with open('rags.json', 'r', encoding='utf-8') as f:
        rags_eval = json.load(f)

    rags_eval_dfs = {rag_name: [create_evaluation_dataset(results) for results in queries_results] for
                     rag_name, queries_results in rags_eval.items()}

    evaluation_results = dict()
    discriminator = openai_model.provide(base_temperature=0.3)[0]
    metrics = ['context_recall', 'faithfulness', 'factual_correctness(mode=f1)']

    for rag_name, eval_list in rags_eval_dfs.items():
        evaluation_results[rag_name] = evaluate_datasets(discriminator, eval_list)

    # for rag_name, eval_res in evaluation_results.items():
    #     print(rag_name)
    #     for q_n, q_stats in eval_res.items():
    #         print(q_n)
    #         print(q_stats[metrics])
    #         print()
    #     print()

    for rag_name, eval_res in evaluation_results.items():
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
    # with open('rags_eval.json', 'w', encoding='utf-8') as f:
    #     json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    # with open('rags_eval.json', 'r', encoding='utf-8') as f:
    #     evaluation_results = json.load(f)


