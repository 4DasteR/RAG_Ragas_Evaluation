import pandas as pd
from typing import List, Dict
from .rag_chain import SimpleRAG
from ragas import EvaluationDataset
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness


def create_evaluation_dicts(rag: SimpleRAG,
                            engineered_queries: List[Dict[str, str]],
                            ground_truth_answers: List[str]
                            ) -> list[list[dict[str, str]]]:
    result_list: list[list[dict]] = []
    for idx, en_query in enumerate(engineered_queries):
        query_result: list[dict] = []
        for name, query in en_query.items():
            result = rag.query(query)
            question = query
            answer = result["answer"]
            context = [doc.page_content for doc in result["source_documents"]]
            ground_truth = ground_truth_answers[idx]
            query_result.append({
                "question": question,
                "answer": answer,
                "context": context,
                "ground_truth": ground_truth
            })
        result_list.append(query_result)
    # Add answers to dataset for rag
    return result_list


def create_evaluation_dataset(eval_dicts: List[Dict[str, str]]) -> EvaluationDataset:
    df = pd.DataFrame(eval_dicts)
    df["retrieved_contexts"] = df["context"]
    df["user_input"] = df["question"]
    df["response"] = df["answer"]
    df["reference"] = df["ground_truth"]

    return EvaluationDataset.from_pandas(df[["user_input", "retrieved_contexts", "response", "reference"]])


def evaluate_datasets(discriminator, eval_list):
    results = dict()
    for i, evaluation_dataset in enumerate(eval_list):
        results[f"Q{i + 1}"] = evaluate(
            dataset=evaluation_dataset,
            metrics=[
                LLMContextRecall(),
                Faithfulness(),
                FactualCorrectness()
            ],
            llm=discriminator,
        ).to_pandas()#.to_dict(orient='records')
    return results
