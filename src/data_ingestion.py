import pandas as pd
from src.logger import setup_logger
import requests
import os

# Setting up the directory
DATA_DIR = './data/raw/'
os.makedirs(DATA_DIR, exist_ok=True)

# Setting up logger
logger = setup_logger(name='data_ingestion', log_file='logs/data_ingestion.log')

def data_ingestion(url: str) -> None:
    """This function downloads a PDF from a given URL and saves it locally."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Check if the content is a PDF
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type.lower():
                file_name = "movies_data.pdf"
                file_path = os.path.join(DATA_DIR, file_name)
                with open(file_path, "wb") as pdf_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            pdf_file.write(chunk)
                logger.debug(f"PDF has been saved at: {file_path}")
            else:
                logger.error("The URL does not point to a PDF file.")
        else:
            logger.error(f"Error occurred while getting the data from the URL. Status code: {response.status_code}")
    except Exception as e:
        logger.error("Unexpected error occurred during data ingestion", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        url = "https://raw.githubusercontent.com/Dharansh-Neema/RAG-movie-info/main/data/top_movies.pdf"
        data_ingestion(url)
    except Exception as e:
        logger.error("Unexpected error occurred in the main execution", exc_info=True)
        raise
