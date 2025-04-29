from langgraph.graph import END, StateGraph
from graph.nodes import questions_generator, response_generator, questions_router, question_evaluator, question_evolver, question_reporter
from graph.graph_state import GraphState

# Create and configure the graph
def create_graph() -> StateGraph:
    """Create the workflow graph."""
    # Define nodes
    workflow = StateGraph(GraphState)

    # Add the question generator node
    workflow.add_node("node_questions_generator", questions_generator)
    workflow.add_node("node_response_generator", response_generator)
    workflow.add_node("node_question_evaluator", question_evaluator)
    workflow.add_node("node_question_evolver", question_evolver)
    workflow.add_node("node_question_reporter", question_reporter)
    workflow.add_node("node_questions_router", questions_router)

    workflow.add_edge("node_questions_generator", "node_questions_router")
    workflow.add_edge("node_response_generator", "node_questions_router")
    workflow.add_edge("node_question_evaluator", "node_questions_router")
    workflow.add_edge("node_question_evolver", "node_questions_router")
    workflow.add_edge("node_question_reporter", "node_questions_router")

    workflow.add_conditional_edges(
        "node_questions_router",
        lambda x: x.next_step,
        {
            "RESPONSE": "node_response_generator",
            "EVALUATE": "node_question_evaluator",
            "EVOLVE": "node_question_evolver",
            "REPORT": "node_question_reporter",
            "DONE": END,
        },
    )

    # Set entry point
    workflow.set_entry_point("node_questions_generator")

    # Compile the graph
    return workflow.compile()
