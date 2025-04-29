from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from models.question import Question
from graph.graph_state import GraphState
from typing import Dict, Any
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
import pandas as pd

GENERATOR_MODEL = "gpt-4o-mini"
RESPONSE_MODEL = "gpt-4o-mini"
EVALUATOR_MODEL = "gpt-4.1-mini"

# Define the question generator node
def questions_generator(state: GraphState) -> Dict[str, Any]:
    """Generate simple questions based on documents in the graph state."""

    PROMPT = """
    CONTEXT:
    {context}

    Generate {num_questions} simple random questions about the context provided. Do not use any other knowledge than the context provided. The generated questions should be easy to answer and must not exceed 8 words. The questions will be separated by newlines and will not be numbered.
    """
    chain_prompt = ChatPromptTemplate.from_template(PROMPT)
    chat_model = ChatOpenAI(model=GENERATOR_MODEL)

    def get_prompt_variables(state: GraphState):
        # Join all document contents into a single string
        context = "\n".join([doc.page_content for doc in state.docs])
        return {"context": context, "num_questions": state.num_questions}

    chain = chain_prompt | chat_model | StrOutputParser()

    try:
        questions_str = chain.invoke(get_prompt_variables(state))
        # Split questions by line or another delimiter as appropriate
        questions_list = [q.strip() for q in questions_str.split('\n') if q.strip()]
        for question_text in questions_list:
            question = Question()
            question.update_question(question_text)
            state.questions.append(question)
        return state
    except Exception as e:
        state.error = str(e)
        return {"error": state.error}

def questions_router(state: GraphState) -> Dict[str, Any]:
    """Route to next node or end based on state."""
    if state.error:
        return state

    # If current_question is not defined or is DONE, find next question
    if not state.current_question or state.current_question.stage == "REPORT":
        for question in state.questions:
            if question.stage != "REPORT":
                state.current_question = question
                break
    
    if state.current_question.stage == "REPORT":
        # If the current question is marked as REPORT, this means all questions 
        # have already been processed and included in the report. Set the next 
        # step to DONE to signal the end of the workflow.
        state.next_step = "DONE"    
    elif state.current_question.stage == "SEED":
        # If the current question is marked as SEED, this means the question
        # has just started the workflow. Set the next step to RESPONSE to 
        # generate a response to the question.
        state.next_step = "RESPONSE"
    elif state.current_question.stage == "RESPONSE":
        # If the current question is marked as RESPONSE, this means the question
        # has been responded to. Set the next step to EVALUATE to evaluate the 
        # question's response.
        state.next_step = "EVALUATE"
    elif state.current_question.stage == "EVALUATE":
        if state.current_question.evaluation.outcome == "accept":
            state.next_step = "EVOLVE"  
        else:
            state.next_step = "REPORT"  
    elif state.current_question.stage == "EVOLVE":
        state.next_step = "REPORT"
    return state

def response_generator(state: GraphState) -> Dict[str, Any]:
    """Generate a response to the question."""

    PROMPT = """
    CONTEXT:
    {context}

    {question}\nAnswer the question as accurately as possible using only the context provided.
    """
    chain_prompt = ChatPromptTemplate.from_template(PROMPT)
    chat_model = ChatOpenAI(model=RESPONSE_MODEL)

    def get_prompt_variables(state: GraphState):
        # Join all document contents into a single string
        context = "\n".join([doc.page_content for doc in state.docs])
        return {"context": context, "question": state.current_question.question_text}

    chain = chain_prompt | chat_model | StrOutputParser()

    try:
        answer = chain.invoke(get_prompt_variables(state))
        state.current_question.update_response(answer)
        state.current_question.update_stage("RESPONSE")
        return state
    except Exception as e:
        state.error = str(e)
        return {"error": state.error}

def question_evaluator(state: GraphState) -> Dict[str, Any]:
    """Evaluate the response to the question."""

    custom_run_config = RunConfig(timeout=360)
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=EVALUATOR_MODEL))
    context_docs = [doc.page_content for doc in state.docs]
    # Prepare the data for evaluation
    dataset = [
        {
            "user_input": state.current_question.question_text,
            "response": state.current_question.response_text,
            "reference_contexts": context_docs,
            "retrieved_contexts": context_docs,
        }
    ]
    evaluation_dataset = EvaluationDataset.from_dict(dataset)

    # Evaluate using RAGAS metrics
    result = evaluate(
        evaluation_dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=evaluator_llm,
        run_config=custom_run_config
    )

    # Update the question with evaluation results
    state.current_question.evaluate_response(
        faithfulness=result['faithfulness'][0],
        response_relevancy=result['answer_relevancy'][0]
    )

    state.current_question.update_stage("EVALUATE")
    return state

def question_evolver(state: GraphState) -> Dict[str, Any]:
    """Create variants of a question."""
    state.current_question.update_stage("EVOLVE")
    return state

def question_reporter(state: GraphState) -> Dict[str, Any]:
    """Generate a report of the question."""
    state.current_question.update_stage("REPORT")
    return state