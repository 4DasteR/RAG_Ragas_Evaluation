from typing import Optional, Dict, Set, Tuple, Literal
from dataclasses import dataclass, field

SEMANTIC_UNITS: Set[str] = frozenset({'paragraphs', 'sentences', 'words'})

@dataclass
class Query:
    prompt_engineered_text: Dict[str, str] = field(default_factory=dict)
    ground_truth_answer: Optional[str] = None

@dataclass
class QueryBuilder():
    __query_text: str = field(default=None, init=False)
    __ground_truth_answer: Optional[str] = field(default=None, init=False)
    __zero_shot: Tuple[bool, int, int, Literal['paragraphs', 'sentences', 'words']] = field(init=False, default_factory=lambda:(
        False, 1, 3, "paragraphs"
    ))
    __chain_of_thought: bool = field(default=False, init=False)
    __role_prompting: Tuple[bool, Optional[str]] = field(init=False, default_factory=lambda: (False, None))
        
    def reset(self):
        self.__query_text = None
        self.__ground_truth_answer = None
        self.__zero_shot = (False, 1, 3, "paragraphs")
        self.__chain_of_thought = False
        self.__role_prompting = (False, None)
    
    def text(self, text: str) -> 'QueryBuilder':
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Query text must be provided as a nonempty string!")
        
        self.__query_text = text
        return self
    
    def zero_shot(self, 
                  min_val: int = 1, 
                  max_val: int = 3, 
                  semantic_unit: Literal['paragraphs', 'sentences', 'words'] = "paragraphs"
                ) -> 'QueryBuilder':
        
        if not isinstance(min_val, int) or min_val < 1:
            raise ValueError("Minimal value must be a non-zero positive integer!")
        
        if not isinstance(max_val, int) or max_val <= min_val:
            raise ValueError("Maximal value must be an integer greater than minimal!")

        if semantic_unit not in SEMANTIC_UNITS:
            raise ValueError(f"Semantic unit must be one of those: {SEMANTIC_UNITS}")
        
        self.__zero_shot = (True, min_val, max_val, semantic_unit)
        return self
    
    def chain_of_thought(self) -> 'QueryBuilder':
        self.__chain_of_thought = True
        return self
    
    def role_prompting(self, role_knowledge: str) -> 'QueryBuilder':
        if len(role_knowledge.strip()) == 0 or type(role_knowledge) is not str:
            raise ValueError("Role must be a nonempty string!")
        
        self.__role_prompting = (True, role_knowledge)
        return self
    
    def ground_truth_answer(self, answer: str) -> 'QueryBuilder':
        if not isinstance(answer, str) or len(answer.strip()) == 0:
            raise ValueError("Ground truth answer must be a nonempty string!")
        
        self.__ground_truth_answer = answer
        return self
    
    def build(self) -> Query:
        if not self.__query_text:
            raise ValueError("Query text must be set before building the query.")

        query_dict = dict()
        
        if self.__zero_shot[0]:
            min, max, semantic_unit = self.__zero_shot[1:]
            query_dict["zero_shot"] = self.__query_text + f"\nExplain in between {min} and {max} {semantic_unit}."
            
        if self.__chain_of_thought:
            query_dict["chain_of_thought"] = self.__query_text + "\nPlease think about this step by step."
            
        if self.__role_prompting[0]:
            query_dict["role_prompting"] = f"You are an expert in {self.__role_prompting[1]}.\n" + self.__query_text
            
        if len(query_dict) == 0:
            query_dict["no_technique"] = self.__query_text
            
        query = Query(query_dict, self.__ground_truth_answer)
        self.reset()
        return query