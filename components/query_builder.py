from dataclasses import dataclass, field
from typing import Optional, Dict, Set, Tuple, Literal, List

from .validation_methods import *

SEMANTIC_UNITS: Tuple[str, ...] = ('paragraphs', 'sentences', 'words')
ALL_TECHNIQUES: Tuple[str, ...] = ('Zero-Shot', 'Chain of Thought', 'Role Prompting', 'Self Consistency', 'Directional Stimulus')


@dataclass(frozen=True)
class Query:
    """
    Class representing a query to RAG chain. Can be created only via QueryBuilder.

    Attributes:
        text (str): Text of the query
        ground_truth_answer(str): Correct answer to the query provided by user. Used for evaluation
        prompt_engineered_text (Dict[str, str]): Dictionary of engineered query, where key is a technique and value is query engineered according to that technique
    """
    text: str
    ground_truth_answer: str
    prompt_engineered_text: Dict[str, str] = field(default_factory=dict)


@dataclass
class QueryBuilder():
    __query_text: str = field(default=None, init=False)
    __ground_truth_answer: Optional[str] = field(default=None, init=False)
    __techniques_used: Set[Literal['zero_shot', 'chain_of_thought', 'role_prompting', 'self_consistency', 'directional_stimulus']] = field(init=False, default_factory=set)
    __zero_shot: Tuple[int, int, Literal['paragraphs', 'sentences', 'words']] = field(init=False, default_factory=lambda: (1, 3, "paragraphs"))
    __role_prompting: Optional[str] = field(init=False, default=None)
    __self_consistency: int = field(init=False, default=2)
    __directional_stimulus: List[str] = field(init=False, default_factory=list)

    def reset(self):
        self.__query_text = None
        self.__ground_truth_answer = None
        self.__zero_shot = (1, 3, "paragraphs")
        self.__role_prompting = None
        self.__self_consistency = 2
        self.__directional_stimulus.clear()
        self.__techniques_used.clear()

    def text(self, text: str) -> 'QueryBuilder':
        if not validate_string(text):
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

        self.__zero_shot = (min_val, max_val, semantic_unit)
        self.__techniques_used.add('zero_shot')
        return self

    def chain_of_thought(self) -> 'QueryBuilder':
        self.__techniques_used.add('chain_of_thought')
        return self

    def role_prompting(self, role_knowledge: str) -> 'QueryBuilder':
        if not validate_string(role_knowledge):
            raise ValueError("Role must be a nonempty string!")

        self.__role_prompting = role_knowledge
        self.__techniques_used.add('role_prompting')
        return self

    def self_consistency(self, n_answers: int = 2) -> 'QueryBuilder':
        if not isinstance(n_answers, int) or n_answers < 2:
            raise ValueError("Need at least 2 answers for self-consistency")

        self.__self_consistency = n_answers
        self.__techniques_used.add('self_consistency')
        return self

    def directional_stimulus(self, hints: List[str]) -> 'QueryBuilder':
        if not isinstance(hints, list) or len(hints) == 0 or any([not validate_string(hint) for hint in hints]):
            raise ValueError("Hints must be a list of valid strings!")

        self.__directional_stimulus = hints
        self.__techniques_used.add('directional_stimulus')
        return self

    def ground_truth_answer(self, answer: str) -> 'QueryBuilder':
        if not validate_string(answer):
            raise ValueError("Ground truth answer must be a nonempty string!")

        self.__ground_truth_answer = answer
        return self

    def build(self) -> Query:
        if not self.__query_text:
            raise ValueError("Query text must be set before building the query.")

        if not self.__ground_truth_answer:
            self.__ground_truth_answer = "No ground truth answer provided."

        query_dict = dict()

        if 'zero_shot' in self.__techniques_used:
            min_, max_, semantic_unit = self.__zero_shot
            query_dict["zero_shot"] = self.__query_text + f"\nExplain in between {min_} and {max_} {semantic_unit}."

        if 'chain_of_thought' in self.__techniques_used:
            query_dict["chain_of_thought"] = self.__query_text + "\nPlease think about this step by step."

        if 'role_prompting' in self.__techniques_used:
            query_dict["role_prompting"] = f"You are an expert in {self.__role_prompting}.\n" + self.__query_text

        if 'self_consistency' in self.__techniques_used:
            query_dict["self_consistency"] = self.__query_text + f"\nPlease generate {self.__self_consistency} answers and select the most consistent one."

        if 'directional_stimulus' in self.__techniques_used:
            query_dict['directional_stimulus'] = self.__query_text + "\nHere are some hints: \n\t-" + '\n\t-'.join(self.__directional_stimulus)

        if len(query_dict) == 0:
            query_dict["no_technique"] = self.__query_text

        query = Query(self.__query_text, self.__ground_truth_answer, query_dict)
        self.reset()
        return query
