from typing import List, Dict, Any, Annotated
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langgraph.graph import StateGraph
from models.question import Question

# Define the state schema
class GraphState(BaseModel):
    """State for the question generation workflow."""
    docs: List[Document] = Field(default_factory=list)
    questions: List[Question] = Field(default_factory=list)
    error: str = ""
    num_questions: int = 1

# Define the question generator node
def question_generator(state: GraphState) -> Dict[str, Any]:
    """Generate questions based on input documents."""
    try:
        # Here you would implement the actual question generation logic
        # For now we'll create placeholder questions
        for _ in range(state.num_questions):
            question = Question()
            question.update_question("Placeholder question")  # Replace with actual generation
            state.questions.append(question)
            
        return {"questions": state.questions}
    except Exception as e:
        state.error = str(e)
        return {"error": state.error}

# Create and configure the graph
def create_graph() -> StateGraph:
    """Create the workflow graph."""
    # Define nodes
    workflow = StateGraph(GraphState)

    # Add the question generator node
    workflow.add_node("QuestionGenerator", question_generator)

    # Define the conditional edges
    def router(state: GraphState) -> str:
        """Route to next node or end based on state."""
        if state.error:
            return "END"
        return "END"

    # Define the conditional edges
    workflow.add_conditional_edges(
        "QuestionGenerator",
        router
    )

    # Set entry point
    workflow.set_entry_point("QuestionGenerator")

    # Compile the graph
    return workflow.compile()
