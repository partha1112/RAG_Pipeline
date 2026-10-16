from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from src.service.ustream_services import fetch_user_data_with_retry
from src.utils.exceptions import PoorRetrievalQualityException
from src.service.input_processor import rewrite_input
from src.service.vectorstore import get_documents_without_metadata
from src.llm import llm
from src.guardrails.validator import vaidate_output, validate_input
from src.utils.schema import UserInput
import asyncio


def process_input(userInput : UserInput):

    user_info = asyncio.run(fetch_user_data_with_retry(userInput.user_id))

    if user_info:
        return user_info
    
    input_query = userInput.message
    validation_result = validate_input(input_query)
    if not validation_result["is_valid"]:
        return f"Your question is {validation_result['status']}. Please rephrase your question and try again."

    rewritten_input = rewrite_input(input_query)
    documents = get_documents_without_metadata(rewritten_input)

    score = documents[0][1]

    if score < 1.5 :
        raise PoorRetrievalQualityException()
    
    prompt_doc = documents[0][0].page_content

    PROMPT_TEMPLATE = f"""You are an accurate and helpful HR Assistant. Your role is to answer employee questions using ONLY the official HR documentation provided below.

    Strict Guidelines:
    1. Base your answer strictly on the facts present in the provided context. Do not assume, extrapolate, or bring in outside knowledge.
    2. If the provided context does not contain enough information to answer the question, clearly state: "I'm sorry, but the provided HR documentation does not contain information to answer your question."
    3. Keep your tone professional, concise, and helpful.

    Context:
    {prompt_doc}
    
    """
    print(prompt_doc)
    messages = [SystemMessage(content=PROMPT_TEMPLATE), HumanMessage(content=input_query)]
    response = llm.invoke(messages)

    val_res = vaidate_output(response.content, prompt_doc, rewritten_input)

    print(val_res)
    return response.content

    

    
