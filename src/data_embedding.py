import os
import tiktoken
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src.logger import setup_logger
from dotenv import load_dotenv
load_dotenv()

class DataEmbedding:
    """
    A class to handle text splitting, embedding generation, and storage in Pinecone DB.
    """
    def __init__(self):
        """
        Initializes the DataEmbedding class with logger, embedding model, Pinecone connection, and text splitter setup.
        """
        self.logger = setup_logger(name="data_embedding")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._initialize_pinecone()
        self.text_splitter = None
        self.id = 1

    def _initialize_pinecone(self):
        """Initializes Pinecone client and ensures the index exists."""
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        self.pinecone = Pinecone(api_key=pinecone_api_key)
        self.index_name = "embeddings-pdf-index"
        
        # Create index if it doesn't exist
        if self.index_name not in self.pinecone.list_indexes().names():
            self.logger.info(f"Creating Pinecone index '{self.index_name}'")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2 model
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        self.index = self.pinecone.Index(self.index_name)

    def split_text(self, text : pd.DataFrame , chunk_size: int = 500, chunk_overlap: int = 50) -> list:
        """
        Splits text into manageable chunks using token-based length function.
        
        Args:
            text: Input text to be split
            chunk_size: Maximum number of tokens per chunk (default: 500)
            chunk_overlap: Number of overlapping tokens between chunks (default: 50)
        
        Returns:
            List of text chunks
        """
        # Initialize tokenizer and length function
        tokenizer = tiktoken.get_encoding("cl100k_base")
        def length_function(text: str) -> int:
            return len(tokenizer.encode(text))
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", " ", ""]
        )
        df = text
        all_chunks=[]
        for content in df["content"].dropna():
            chunks = self.text_splitter.split_text(content)
            all_chunks.append(chunks)
        self.logger.info(f"Split text into {len(chunks)} chunks")
        
        # Optionally save chunks to disk
        self._save_chunks_to_disk(chunks)
        return all_chunks

    def _save_chunks_to_disk(self, chunks: list):
        """Saves text chunks to ./data/chunks.txt for debugging/auditing purposes."""
        if not os.path.exists('./data'):
            os.makedirs('./data')
        
        with open('./data/chunks.txt', 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(f"{chunk}\n\n")
        self.logger.info("Saved text chunks to ./data/chunks.txt")

    def embed_and_store(self, chunks: list):
        """
        Generates embeddings for text chunks and stores them in Pinecone.
        
        Args:
            chunks: List of text chunks to be embedded and stored
        """
        # Generate embeddings
        embeddings = self.model.encode(chunks, convert_to_tensor=False)
        
        # Prepare vectors for Pinecone
        vectors = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                'id': str(self.id),
                'values': embedding.tolist(),
                'metadata': {'text': chunk}
            })
        self.id+=1
        # Upsert vectors in batches (Pinecone supports up to 100 vectors per upsert)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        self.logger.info(f"Successfully stored {len(vectors)} vectors in Pinecone")

if __name__ == "__main__":
    # Initialize embedding system
    embedder = DataEmbedding()
    
    # Example text (replace with your actual text)
    doc = pd.read_csv("./data/processed/processed_movie_data.csv")
    
    # Split text into chunks
    chunks = embedder.split_text(doc, chunk_size=500, chunk_overlap=50)
    
    # Generate embeddings and store in Pinecone
    embedder.embed_and_store(chunks)