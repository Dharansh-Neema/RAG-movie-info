import os
from typing import List, Optional
import openai
from src.logger import setup_logger
from langchain_openai import ChatOpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

logger = setup_logger(name="retrieval")

class Retrieval:
    """Enhanced class for robust RAG implementation with improved error handling and type safety."""

    def __init__(self):
        self._validate_environment()
        self.llm = self._initialize_llm()
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _validate_environment(self) -> None:
        """Validate required environment variables."""
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        if not self.openai_key:
            logger.critical("OPENAI_API_KEY not found in environment")
            raise EnvironmentError("OPENAI_API_KEY environment variable required")
        if not self.pinecone_key:
            logger.critical("PINECONE_API_KEY not found in environment")
            raise EnvironmentError("PINECONE_API_KEY environment variable required")

    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize and configure the language model."""
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self.openai_key,
            temperature=0.7,
            max_tokens=500
        )

    def _get_index(self):
        """Retrieve Pinecone index with enhanced error handling."""
        try:
            index = self.pc.Index("rag-for-movies")
            logger.info("Successfully connected to Pinecone index")
            return index
        except Exception as e:
            logger.exception("Failed to retrieve Pinecone index")
            raise ConnectionError("Failed to connect to vector database") from e

    def _process_query_results(self, results: dict) -> List[str]:
        """Safely process and validate Pinecone query results."""
        context_data = []
        
        if not results.get("matches"):
            logger.warning("No matches found in Pinecone response")
            return context_data

        for match in results["matches"]:
            try:
                metadata = match.get("metadata", {})
                text_data = metadata.get("text", "")

                if isinstance(text_data, list):
                    text_data = " ".join(map(str, text_data))
                elif not isinstance(text_data, str):
                    text_data = str(text_data)

                if text_data.strip():
                    context_data.append(text_data.strip())

            except Exception as e:
                logger.warning(f"Error processing match entry: {e}")
                continue

        return context_data

    def generate_context(self, query: str) -> str:
        """Generate context from vector store with robust data validation."""
        try:
            index = self._get_index()
            embedded_query = self.embedding_model.encode(
                query, 
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            query_response = index.query(
                vector=embedded_query.tolist(),
                include_metadata=True,
                top_k=3,
            )

            context_list = self._process_query_results(query_response)
            
            if not context_list:
                logger.info("No valid context retrieved from query")
                return ""

            return "\n".join(context_list)[:4000]  # Ensure token limit safety

        except Exception as e:
            logger.exception("Context generation failed")
            raise RuntimeError("Failed to generate context") from e

    def generate_llm_prompt(self, query: str) -> str:
        """Construct LLM prompt with validated context."""
        try:
            context = self.generate_context(query)
            
            prompt_template = (
                "Respond using only the following context. "
                "If context is unavailable or irrelevant, respond with: "
                "'No relevant information available.'\n\n"
                "Context:\n{context}\n\n"
                "Question: {query}\n\n"
                "Provide a concise, accurate response:"
            )
            
            return prompt_template.format(
                context=context if context else "No context available",
                query=query
            )
            
        except Exception as e:
            logger.exception("Prompt generation failed")
            raise RuntimeError("Failed to construct LLM prompt") from e

    def llm_response(self, query: str) -> str:
        """Generate final LLM response with comprehensive error handling."""
        try:
            prompt = self.generate_llm_prompt(query)
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.exception("LLM response generation failed")
            return "Error generating response. Please try again later."

if __name__ == '__main__':
    try:
        retrieval = Retrieval()
        sample_query ="""2.GodFather rating?
                        Rate
                        4. GodFatherII rating? 
                        out of 10"""
        response = retrieval.llm_response(query=sample_query)
        print(f"Response: {response}")
    except Exception as e:
        logger.critical(f"Application failed: {e}")
        print("Critical system error. Check logs for details.")