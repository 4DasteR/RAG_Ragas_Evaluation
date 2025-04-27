import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from langchain_openai import ChatOpenAI
from pandas import DataFrame
from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

from .logger import Logger
from .query_builder import Query
from .rag_chain import RAG

logger = Logger()

@dataclass
class Evaluator:
    """
    Class representing an evaluator for a query to the RAG.

    Attributes:
        rag (RAG): reference to RAG system
        discriminator (ChatOpenAI): llm that will evaluate the query
        engineered_queries(List[Query]): list of engineered queries for evaluation
    """
    rag: RAG
    discriminator: ChatOpenAI
    engineered_queries: List[Query] = field(default_factory=list)
    __results_folder: Path = Path("results")
    __evaluation_results: Optional[Dict[str, DataFrame]] = None
    __evaluation_list: Optional[List[List[Dict[str, str]]]] = field(default=None, init=False)
    __evaluation_datasets: Optional[List[EvaluationDataset]] = field(default=None, init=False)
    
    def __post_init__(self):
        if not isinstance(self.rag, RAG):
            raise ValueError("RAG system must be provided!")

        if not isinstance(self.engineered_queries, List) or any([not isinstance(q, Query) for q in self.engineered_queries]):
            raise ValueError("Engineered queries must be provided as a list of Query objects!")

        self.__results_folder.mkdir(exist_ok=True)
        logger.log(f"Evaluator for {type(self.rag).__name__} created.", "completed")
    
    @property
    def evaluation_list(self) -> List[List[Dict[str, str]]]:
        """
        Returns a list of prepared queries for later use in evaluator
        """
        if self.__evaluation_list is None:
            self.__evaluation_list = []
            for idx, query in enumerate(self.engineered_queries):
                en_query: Dict[str, str] = query.prompt_engineered_text
                query_result: List[Dict[str, str]] = []
                for technique, query_str in en_query.items():
                    result = self.rag.query(query_str)
                    question = query_str
                    answer = result["answer"]
                    context = [doc.page_content for doc in result["source_documents"]]
                    ground_truth = query.ground_truth_answer
                    query_result.append({
                        "question": question,
                        "answer": answer,
                        "context": context,
                        "ground_truth": ground_truth
                    })
                self.__evaluation_list.append(query_result)
        return self.__evaluation_list

    @property
    def evaluation_datasets(self) -> List[EvaluationDataset]:
        """
        Converts evaluation list into list of EvaluationDatasets
        """
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
        """
        Evaluates provided queries according to RAGAS metrics

        Returns:
            Dict[str, DataFrame]: key is a number of question in following pattern: Qn and value is pandas dataframe containing the values for EvaluationDataset enriched by the evaluated metrics values.
        """
        logger.log(f"Evaluating dataset for {type(self.rag).__name__}...", "EVALUATION")
        results = dict()
        for i, evaluation_dataset in enumerate(self.evaluation_datasets):
            logger.log(f"Evaluating Q{i + 1} for {type(self.rag).__name__}...", "QUERY")
            results[f"Q{i + 1}"] = evaluate(
                dataset=evaluation_dataset,
                metrics=[
                    LLMContextRecall(),
                    Faithfulness(),
                    FactualCorrectness()
                ],
                llm=self.discriminator,
                show_progress=False,
            ).to_pandas()
        self.__evaluation_results = results
        logger.log("Evaluation complete.", "COMPLETED")
        return results
    
    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        if not self.__evaluation_results:
            raise ValueError("Evaluation results must be computed!")

        return {q: df.to_dict(orient='records') for q, df in self.__evaluation_results.items()}
    
    def save_to_json(self, path: Optional[str] = None):
        if not path:
            path = f"{self.__results_folder}/{type(self.rag).__name__}_results.json"
        
        logger.log(f"Saving evaluation results for {type(self.rag).__name__}.", "SAVING")
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self.to_dict(), file, indent=4)