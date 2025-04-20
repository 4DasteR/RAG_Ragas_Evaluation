import pandas as pd
from typing import List, Dict
from .rag_chain import RAG
from .logger import Logger
from ragas import EvaluationDataset
from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from dataclasses import dataclass, field
from typing import Optional
from pandas import DataFrame
from langchain_openai import ChatOpenAI
import json
from pathlib import Path

logger = Logger()

@dataclass
class Evaluator:
    rag: RAG
    discriminator: ChatOpenAI
    engineered_queries: List[Dict[str, str]] = field(default_factory=list)
    ground_truth_answers: List[str] = field(default_factory=list)
    __results_folder: Path = Path("results")
    __evaluation_results: Optional[Dict[str, DataFrame]] = None
    __evaluation_list: Optional[List[Dict[str, str]]] = field(default=None, init=False)
    __evaluation_datasets: Optional[List[EvaluationDataset]] = field(default=None, init=False)
    
    def __post_init__(self):
        self.__results_folder.mkdir(exist_ok=True)
        logger.log(f"Evaluator for {type(self.rag).__name__} created.")
    
    @property
    def evaluation_list(self) -> List:
        if self.__evaluation_list is None:
            self.__evaluation_list = []
            for idx, en_query in enumerate(self.engineered_queries):
                query_result: list[dict] = []
                for name, query in en_query.items():
                    result = self.rag.query(query)
                    question = query
                    answer = result["answer"]
                    context = [doc.page_content for doc in result["source_documents"]]
                    ground_truth = self.ground_truth_answers[idx]
                    query_result.append({
                        "question": question,
                        "answer": answer,
                        "context": context,
                        "ground_truth": ground_truth
                    })
                self.__evaluation_list.append(query_result)
        return self.__evaluation_list

    @property
    def evaluation_datasets(self) -> List:
        if self.__evaluation_datasets is None:
            self.__evaluation_datasets = []
            for result in self.evaluation_list:
                df = pd.DataFrame(result)
                df["retrieved_contexts"] = df["context"]
                df["user_input"] = df["question"]
                df["response"] = df["answer"]
                df["reference"] = df["ground_truth"]
                self.__evaluation_datasets.append(EvaluationDataset.from_pandas(df[["user_input", "retrieved_contexts", "response", "reference"]]))
        return self.__evaluation_datasets
    
    @property
    def evaluation_results(self):
        if not self.__evaluation_results:
            self.evaluate()
        return self.__evaluation_results

    def evaluate(self) -> Dict[str, DataFrame]:
        logger.log(f"Evaluating dataset for {type(self.rag).__name__}...")
        results = dict()
        for i, evaluation_dataset in enumerate(self.evaluation_datasets):
            logger.log(f"Evaluating Q{i + 1}...")
            results[f"Q{i + 1}"] = evaluate(
                dataset=evaluation_dataset,
                metrics=[
                    LLMContextRecall(),
                    Faithfulness(),
                    FactualCorrectness()
                ],
                llm=self.discriminator,
                show_progress = False,
            ).to_pandas()
        self.__evaluation_results = results
        logger.log("Evaluation complete.")
        return results
    
    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        if not self.__evaluation_results:
            raise ValueError("Evaluation results must be computed!")

        return {q: df.to_dict(orient='records') for q, df in self.__evaluation_results.items()}
    
    def save_to_json(self, path: Optional[str] = None):
        if not path:
            path = f"{self.__results_folder}/{type(self.rag).__name__}_results.json"
        
        logger.log(f"Saving evalution results for {type(self.rag).__name__}.")
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self.to_dict(), file, indent=4)