import os 
import pymupdf as pdf
import pandas
from src.logger import setup_logger

logger = setup_logger(name="data_processing")
DATA_DIR = "./data/"
os.makedirs(data_dir,exist_ok=True)
class DataPreprocessing:
    
    
    def remove_spaces(text:str):
        """Remove spaces and lines"""
        try:
            cleaned_text = text.replace("/n","").strip()
            return cleaned_text
        except Exception as e:
            logger.error("Some execption occured while cleaning the data",e)
            raise

    def read_pdf(data_path:str):
        try:
            doc = pdf.open(data_path)
            page_text = []

            for page_number, page in enumerate(doc):
                text = page.get_text().encode("utf-8")
                text = remove_spaces(text)
                page_text.append({
                    "page_number":page_number,
                    "page_char_count":len(text)
                })
            logger.debug("PDF read successfully")
            return page_text
        except Exception as e:
            logger.error("Error occured while pre-processing",e)
            raise

    def save_data(data,file_name:str)->None:
        try:
            file_path = f"processed/{file_name}"
            file_path = os.path.join(DATA_DIR,file_path)
            pd.save_csv(data,file_path)
            logger.debug("PDF stored successfully")
        except Exception as e:
            logger.error("Some execption occur while saving data",e)
            raise

if __name__ == '__main__':
    
    file_path = os.join.path(DATA_DIR,"raw/movies_data.pdf")
    data = DataPreprocessing.read_pdf(file_path)
    DataPreprocessing.save_data(data,"processed_movie_data.csv")

