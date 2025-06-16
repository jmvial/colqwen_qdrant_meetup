from pdf2image import convert_from_path
import os
from dotenv import load_dotenv

load_dotenv("colqwen_qdrant_meetup/.env")

PDF_NAME = os.getenv("PDF_NAME")
DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

pdf_path = DATA_FOLDER_PATH + "/" + PDF_NAME

os.makedirs(DATA_FOLDER_PATH, exist_ok=True)

pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]

pages = convert_from_path(pdf_path, dpi=200)

for i, page in enumerate(pages, start=1):
    filename = f"{pdf_base_name}_{i}.png"
    page.save(os.path.join(DATA_FOLDER_PATH, filename), "PNG")
