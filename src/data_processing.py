import os
import fitz  
import pandas as pd
from src.logger import setup_logger

logger = setup_logger(name="data_processing")

# Define the data directory and ensure it exists
DATA_DIR = "./data/"
os.makedirs(DATA_DIR, exist_ok=True)


class DataPreprocessing:
    
    @staticmethod
    def remove_spaces(text: str) -> str:
        """Remove newline characters and extra spaces from text."""
        try:
            cleaned_text = text.replace("\n", "").strip()
            return cleaned_text
        except Exception as e:
            logger.error(f"Exception occurred while cleaning the data: {e}")
            raise

    @staticmethod
    def read_pdf(data_path: str):
        """Reads a PDF file and extracts text along with metadata."""
        try:
            doc = fitz.open(data_path)
            page_text = []

            for page_number, page in enumerate(doc):
                text = page.get_text("text")
                cleaned_text = DataPreprocessing.remove_spaces(text)

                page_text.append({
                    "page_number": page_number + 1,
                    "page_char_count": len(cleaned_text),
                    "content": cleaned_text
                })

            logger.info(f"Successfully processed PDF: {data_path}")
            return page_text

        except Exception as e:
            logger.error(f"Error occurred while reading the PDF: {e}")
            raise

    @staticmethod
    def save_data(data, file_name: str) -> None:
        """Saves processed data to a CSV file."""
        try:
            file_path = os.path.join(DATA_DIR, "processed", file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)

            logger.info(f"Processed data saved successfully at {file_path}")

        except Exception as e:
            logger.error(f"Exception occurred while saving data: {e}")
            raise


if __name__ == "__main__":
    file_path = os.path.join(DATA_DIR, "raw", "movies_data.pdf")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
    else:
        data = DataPreprocessing.read_pdf(file_path)
        DataPreprocessing.save_data(data, "processed_movie_data.csv")
