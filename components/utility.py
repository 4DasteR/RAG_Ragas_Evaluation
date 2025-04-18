import pandas as pd

def build_rag_metric_dataframes(evaluation_results: dict) -> dict:
    """
    Converts EvaluationResult objects into a dictionary of DataFrames per RAG.

    Args:
        evaluation_results (dict): {rag_name: {"Q1": EvaluationResult, ...}, ...}

    Returns:
        dict: {rag_name: DataFrame of metrics}
    """
    rag_metric_dataframes = {}

    for rag_name, question_results in evaluation_results.items():
        metric_data = {}

        for question, eval_result in question_results.items():
            # Convert EvaluationResult to DataFrame
            df = eval_result.to_pandas()

            for _, row in df.iterrows():
                metric_name = row["metric"]
                score = row["score"]

                if metric_name not in metric_data:
                    metric_data[metric_name] = {}
                metric_data[metric_name][question] = score

        # Build final DataFrame
        df_metrics = pd.DataFrame.from_dict(metric_data, orient='index')
        df_metrics.index.name = "Metric"
        df_metrics = df_metrics.reset_index()

        question_columns = sorted([col for col in df_metrics.columns if col.startswith("Q")])
        df_metrics = df_metrics[["Metric"] + question_columns]

        rag_metric_dataframes[rag_name] = df_metrics

    return rag_metric_dataframes

def print_rag_metric_dataframes(rag_metric_dataframes: dict):
    """
    Nicely prints each RAG's name and its corresponding metrics DataFrame.
    """
    for rag_name, df in rag_metric_dataframes.items():
        print(f"\n=== Metrics for {rag_name} ===")
        print(df.to_string(index=False))
