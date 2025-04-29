from langgraph.graph import END, StateGraph
from graph.nodes import questions_generator, questions_router
from graph.graph_state import GraphState

# Create and configure the graph
def create_graph() -> StateGraph:
    """Create the workflow graph."""
    # Define nodes
    workflow = StateGraph(GraphState)

    # Add the question generator node
    workflow.add_node("node_questions_generator", questions_generator)
    workflow.add_node("node_questions_router", questions_router)

    # Define the conditional edges
    workflow.add_edge("node_questions_generator", "node_questions_router")

    workflow.add_conditional_edges(
        "node_questions_router",
        lambda x: x.next_step,
        {
            "DONE": END,
        },
    )

    # Set entry point
    workflow.set_entry_point("node_questions_generator")

    # Compile the graph
    return workflow.compile()
