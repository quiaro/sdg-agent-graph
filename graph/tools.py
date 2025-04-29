from langchain_core.output_parsers import StrOutputParser
from models.question import Question
from typing import Annotated, List
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from graph.graph_state import GraphState

EVOLVER_MODEL = "gpt-4o-mini"

@tool
def in_depth_deepen(
    question: Annotated[Question, "Question to deepen."],
    context: Annotated[List[Document], "Context to use for deepening."],
) -> Annotated[Question, "Deepened question."]:
    """Create an in-depth variant of a question by going deeper into the subject."""
    
    PROMPT = """
    Add complexity to the following question by adding no more than 6 words.\n\n
    {question}\n\n
    Use only the context provided below to create the question variant.\n\n
    CONTEXT:\n
    {context}\n\n
    Make sure there's only one returned question. Remove any formatting from the question text.
    """
    chain_prompt = ChatPromptTemplate.from_template(PROMPT)
    chat_model = ChatOpenAI(model=EVOLVER_MODEL)

    def get_prompt_variables(docs: List[Document], question: Question):
        # Join all document contents into a single string
        context = "\n".join([doc.page_content for doc in docs])
        return {"context": context, "question": question.question_text}

    chain = chain_prompt | chat_model | StrOutputParser()

    try:
        return chain.invoke(get_prompt_variables(context, question))
    except Exception as e:
        state.error = str(e)
        return {"error": state.error}