import pandas as pd
from src.logger import setup_logger
import requests
import os

#setupig the directory
DATA_DIR = './data/raw'
os.makedirs(DATA_DIR,exist_ok=True)

#setup logger 
logger = setup_logger(name='data_ingestion',log_file='logs/data_ingestion.log')