from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Literal
from datetime import datetime
from uuid import uuid4

class Evolution(BaseModel):
    round: int = 0
    evolver_type: Literal["NONE", "in_depth", "in_breadth"] = "NONE"
    operation: Literal[
        "NONE", "deepen", "add_constraint", "add_reasoning", "complicate_input", "diversity"
    ] = "NONE"
    parent_id: str = "NONE"
    children: List[str] = Field(default_factory=list)

    @field_validator('round')
    @classmethod
    def validate_round(cls, v):
        if v < 0:
            raise ValueError('evolution.round must be >= 0')
        return v

class Difficulty(BaseModel):
    level: Literal["NONE", "easy", "medium", "hard"] = "NONE"
    score: float = 1.0

    @field_validator('score')
    @classmethod
    def validate_score(cls, v):
        if not (1.0 <= v <= 10.0):
            raise ValueError('difficulty.score must be between 1.0 and 10.0')
        return v

class Evaluation(BaseModel):
    complexity_added: bool = False
    faithfulness: float = 0.0
    response_relevancy: float = 0.0
    outcome: Literal["NONE", "accept", "reject"] = "NONE"

    @field_validator('faithfulness')
    @classmethod
    def validate_faithfulness(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('evaluation.faithfulness must be between 0.0 and 1.0')
        return v

    @field_validator('response_relevancy')
    @classmethod
    def validate_response_relevancy(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('evaluation.response_relevancy must be between 0.0 and 1.0')
        return v

class Metadata(BaseModel):
    source_doc_ids: List[str] = Field(default_factory=list)
    creation_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class HistoryEvent(BaseModel):
    event: str  # stage name (SEED, EVOLVE, etc.)
    timestamp: str

class Question(BaseModel):
    model_config = ConfigDict(extra='forbid')  # disallow unknown fields

    # Configurable thresholds
    faithfulness_threshold: float = 0.8  # default 80%
    relevancy_threshold: float = 0.8     # default 80%

    stage: Literal["SEED", "ASSESS", "EVOLVE", "RESPONSE", "EVALUATE", "DONE"] = "SEED"
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question_text: str = ""
    response_text: str = ""
    evolution: Evolution = Field(default_factory=Evolution)
    difficulty: Difficulty = Field(default_factory=Difficulty)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    metadata: Metadata = Field(default_factory=Metadata)
    history: List[HistoryEvent] = Field(default_factory=list)

    def evaluate_complexity(self, complexity_added: bool):
        """Evaluate whether complexity was added to the question."""
        self.evaluation.complexity_added = complexity_added
        self.evaluation.outcome = "reject" if not complexity_added else "accept"
        self.record_event("EVALUATE_COMPLEXITY")

    def evaluate_response(self, faithfulness: float, response_relevancy: float):
        """Record evaluation metrics and final outcome."""
        self.evaluation.faithfulness = faithfulness
        self.evaluation.response_relevancy = response_relevancy
        self.evaluation.outcome = "accept" if (
            faithfulness > self.faithfulness_threshold
            and response_relevancy > self.relevancy_threshold
        ) else "reject"
        self.record_event("EVALUATE_RESPONSE")

    def finalize(self):
        """Mark as DONE."""
        self.stage = "DONE"
        self.record_event("DONE")

    def initialize_evolution(self, round: int, parent_id: str):
        """Initialize evolution metadata."""
        self.evolution.round = round
        self.evolution.parent_id = parent_id
        self.record_event("INITIALIZE_EVOLUTION")

    def record_event(self, event: str):
        """Record a lifecycle event with timestamp."""
        self.history.append(HistoryEvent(
            event=event,
            timestamp=datetime.utcnow().isoformat()
        ))
    
    def update_difficulty(self, level: str, score: float):
        """Update difficulty level and score."""
        self.difficulty.level = level
        self.difficulty.score = score
        self.record_event("UPDATE_DIFFICULTY")

    def update_question(self, question_text: str):
        """Attach a question."""
        self.question_text = question_text
        self.record_event("UPDATE_QUESTION")

    def update_evolved_question(self, question_text: str, evolver_type: Literal["in_depth", "in_breadth"], operation: Literal["deepen", "add_constraint", "add_reasoning", "complicate_input", "diversity"]):
        """Attach an evolved question."""
        self.question_text = question_text
        self.evolution.evolver_type = evolver_type
        self.evolution.operation = operation
        self.record_event("UPDATE_EVOLVED_QUESTION")

    def update_response(self, response_text: str):
        """Attach a generated response."""
        self.response_text = response_text
        self.record_event("UPDATE_RESPONSE")

