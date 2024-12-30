# RAG Movie Info

This repository demonstrates the implementation of a Retrieval-Augmented Generation (RAG) model applied to a movie database. The RAG model efficiently retrieves information from a vector database and uses a language model to generate responses, making it ideal for querying structured and unstructured datasets.

## Introduction to RAG

Retrieval-Augmented Generation (RAG) is a hybrid approach that combines the strengths of information retrieval systems and language models. The pipeline involves Three key components:

1. **Embedding & Storing** : Embeded the DataSet and store into pinecone
2. **Retriever**: Fetches relevant information from pinecone vector database.
3. **Generator**: Processes the retrieved data and generates a coherent, context-aware response using LLM(openai).

### RAG Pipeline

Below is an overview of the RAG pipeline:

![RAG Pipeline](https://github.com/Dharansh-Neema/RAG-movie-info/blob/main/utils/RAG1.png)

## Tech Stack

The following technologies were used to build the RAG Movie Info system:

- **LangChain**: A framework for building applications powered by language models.
- **OpenAI**: Provides the LLM (e.g., GPT-3.5-Turbo) for generating responses.
- **Hugging Face**: Used for embedding text data into vector representations.
- **Pinecone**: Serves as the vector database for efficient similarity search and retrieval.

## Installation Steps

Follow these steps to set up the project locally:

### Step 1: Clone the Repository

```bash
git clone https://github.com/Dharansh-Neema/RAG-movie-info.git
cd RAG-movie-info
```

### Step 2: Create a Virtual Environment

Create a virtual environment to isolate the dependencies for this project:

```bash
python -m venv env
```

### Step 3: Activate the Virtual Environment

- On **Windows**:
  ```bash
  .\env\Scripts\activate
  ```
- On **Linux/MacOS**:
  ```bash
  source env/bin/activate
  ```

### Step 4: Install Dependencies

Install all required dependencies listed in `dependency.txt`:

```bash
pip install -r dependency.txt
```

### Step 5: Configure Environment Variables

Create a `.env` file in the root directory and add the following keys:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Replace `your_openai_api_key` and `your_pinecone_api_key` with your actual API keys.
