import openai
from src.logger import logger
import os
from langchain_openai import ChatOpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer
class Retrieval:
    def __init__(self):
        self.__OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")
        self.__PINECONE_API = os.getenv("PINECONE_API") 
        self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=self.__OPEN_AI_KEY,
                temperature=0.7,         
                max_tokens=500)   

        self.pc = Pinecone(api_key=self.__PINECONE_API)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def _get_index(self):
        try:
            index = self.pc("https://embeddings-pdf-index-gfb3fgs.svc.aped-4627-b74a.pinecone.io")
            logger.debug("Fetched index for retrieval")
            return index
        except Exception as e:
            logger.error("Error occurred while fetching the pinecone index".e)
            raise
        
    def get_context(self,query:str):
        try:
            index = self._get_index()
            embedded_query = self.model.encode(query,convert_to_tensor=False)
            logger.debug("Query Embedded Successfully!!")

            results = index.query(
                vector=embedded_query,
                include_metadata = True,
                top_k = 3
            )

            context = [match["metadata"]["text"] for match in results["matches"]]
            logger.debug("Successfully get the context")
        except Exception as e:
            logger.error("Error occured while fethcing the context",e)

        
    
