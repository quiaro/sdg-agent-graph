from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from models.question import Question
from graph.graph_state import GraphState
from typing import Dict, Any

GENERATOR_MODEL = "gpt-4o-mini"

# Define the question generator node
def questions_generator(state: GraphState) -> Dict[str, Any]:
    """Generate simple questions based on documents in the graph state."""

    PROMPT = """
    CONTEXT:
    {context}

    Generate {num_questions} simple random questions about the context provided. Do not use any other knowledge than the context provided. The generated questions should be easy to answer and must not exceed 8 words. The questions will be separated by newlines and will not be numbered.
    """
    chain_prompt = ChatPromptTemplate.from_template(PROMPT)
    openai_chat_model = ChatOpenAI(model=GENERATOR_MODEL)

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

def questions_router(state: GraphState) -> Dict[str, Any]:
    """Route to next node or end based on state."""
    if state.error:
        return state

    # If current_question is not defined or is DONE, find next question
    if not state.current_question or state.current_question.stage == "DONE":
        for question in state.questions:
            if question.stage != "DONE":
                state.current_question = question
                break
    
    state.next_step = "DONE"

    return state
