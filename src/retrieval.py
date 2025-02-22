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
        
    def generate_context(self,query:str)->str:
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
            logger.debug("Successfully got the context")
            return context
        except Exception as e:
            logger.error("Error occured while fethcing the context",e)


    def generate_llm_prompt(self,query:str)->str:
        """
        This function will return a prompt for LLM to get the proper Response.
        """
        try:
            context = self.get_context(query)
            prompt = f"""
            Use the following context to answer the question accurately.SInce you are used in RAG
            If the information is not available in the context, reply "No relevant responses."

            Context:
            {context}

            Question:
            {query}

            Answer:
            """
            logger.debug("Generated prompt for LLM")
        except Exception as e:
            logger.error("Error occurred while generating prompt for LLM")
            raise

    def llm_response(self,query:str)->str:
        """
        This will generate a response form LLM
        """
        try:
            prompt = self.generate_llm_prompt(query=query)
            response = self.llm(prompt)
            logger.debug("Got the response from LLM")
            return response
        except Exception as e:
            logger.error("Error occurred while generating response from LLM")

if __name__ == '__main__':
    query = """2.GodFather rating out of 10"""
    retrieval = Retrieval()

    response = retrieval.llm_response(query=query)
    print(response)



        
    
