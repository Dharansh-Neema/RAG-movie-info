import openai
from src.logger import logger
import os
from langchain_openai import ChatOpenAI
class Retrieval:
    _OPEN_AI_KEY = os.getenv("OPENAI_API_KEY") 
    def __init__(self):
        llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=OPEN_AI_KEY,
                temperature=0.7,         
                max_tokens=500)   
response = llm(prompt)
print(response)
