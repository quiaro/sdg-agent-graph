from typing import List, Dict, Any, Annotated
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
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
    PROMPT = """
    CONTEXT:
    {context}

    Generate {num_questions} simple random questions about the context provided. Do not use any other knowledge than the context provided. The generated questions should be easy to answer and must not exceed 8 words. The questions will be separated by newlines and will not be numbered.
    """
    chain_prompt = ChatPromptTemplate.from_template(PROMPT)
    openai_chat_model = ChatOpenAI(model="gpt-4o-mini")

    def get_prompt_variables(state: GraphState):
        # Join all document contents into a single string
        context = "\n".join([doc.page_content for doc in state.docs])
        return {"context": context, "num_questions": state.num_questions}

    chain = chain_prompt | openai_chat_model | StrOutputParser()

    try:
        questions_str = chain.invoke(get_prompt_variables(state))
        # Split questions by line or another delimiter as appropriate
        questions_list = [q.strip() for q in questions_str.split('\n') if q.strip()]
        for question_text in questions_list:
            question = Question()
            question.update_question(question_text)
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
