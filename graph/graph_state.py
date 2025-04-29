from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from models.question import Question

class GraphState(BaseModel):
    docs: List[Document] = Field(default_factory=list)
    questions: List[Question] = Field(default_factory=list)
    current_question: Question = Field(default_factory=Question)
    num_questions: int = 1
    next_step: Literal["SEED", "RESPONSE", "EVALUATE_RESPONSE", "EVOLVE", "EVALUATE_COMPLEXITY", "REPORT", "DONE"] = "RESPONSE"
    error: str = ""