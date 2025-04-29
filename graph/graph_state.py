from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from models.question import Question

class GraphState(BaseModel):
    docs: List[Document] = Field(default_factory=list)
    questions: List[Question] = Field(default_factory=list)
    error: str = ""
    num_questions: int = 1 