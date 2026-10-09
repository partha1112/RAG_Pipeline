from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from src.llm import llm


def rewrite_input(input_query: str) -> str:
    prompt=  """
    You are a useful we writting agent. You will be given a query and you have to rewrite it in a way that is more specific and relevant to the HR domain. 
    
    Rules: 
    1. If the query is already specific and relevant to the HR domain, return it as is.
    2. If the query is vague or not relevant to the HR domain, rewrite it to be more specific and relevant to the HR domain.

    """
    messages = [SystemMessage(content=prompt), HumanMessage(content=input_query)]
    rewritten_query = llm.invoke(messages)
    print("rewritten query", rewritten_query.content)
    return rewritten_query.content

    

    

        