from typing import Dict, Any, Optional

from components.query_builder import QueryBuilder, Query


def create_query(text: str, ground_truth: Optional[str], techniques: Dict[str, Any]) -> Query:
    query_builder = QueryBuilder().text(text)
    
    if isinstance(ground_truth, str) and len(ground_truth.strip()) != 0:
        query_builder = query_builder.ground_truth_answer(ground_truth)
        
    for technique, params in techniques.items():
        if 'zero_shot' == technique:
            query_builder = query_builder.zero_shot(*params)
            
        elif 'chain_of_thought' == technique:
            query_builder = query_builder.chain_of_thought()
            
        elif 'role_prompting' == technique:
            query_builder = query_builder.role_prompting(params)
            
        elif 'self_consistency' == technique:
            query_builder = query_builder.self_consistency(params)
            
        elif 'directional_stimulus' == technique:
            query_builder = query_builder.directional_stimulus(params)
    
    return query_builder.build()

def evaluate_query():
    pass