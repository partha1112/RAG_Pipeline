from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

llm = ChatOpenAI(model_name="gpt-4o-mini",temperature=0)


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=768,  # As my documet size is less than 30 pages and each document is clearly defined under sub header I have uses 768 dimentions (defailt 1536) 
)