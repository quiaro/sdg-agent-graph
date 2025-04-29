from langgraph.graph import StateGraph
from graph.nodes import questions_generator
from graph.graph_state import GraphState

# Create and configure the graph
def create_graph() -> StateGraph:
    """Create the workflow graph."""
    # Define nodes
    workflow = StateGraph(GraphState)

    # Add the question generator node
    workflow.add_node("QuestionGenerator", questions_generator)

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
