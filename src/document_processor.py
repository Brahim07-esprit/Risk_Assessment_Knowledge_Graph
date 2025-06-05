import re
from typing import List, Tuple

import PyPDF2
import spacy
from Crypto.Cipher import AES

from .config import settings


class DocumentProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def process_document(self, file_path: str) -> Tuple[str, List[str]]:
        text = self.extract_text_from_pdf(file_path)

        text = self.clean_text(text)

        chunks = self.create_chunks(text, chunk_size=settings.MAX_CHUNK_SIZE)

        return text, chunks

    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ""

        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"

        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")

        return text

    def clean_text(self, text: str) -> str:
        text = " ".join(text.split())

        text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')

        text = "".join(char for char in text if ord(
            char) >= 32 or char == "\n")

        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

        patterns = [
            (r"\bfi\s+rst\b", "first"),
            (r"\bidentifi\s+cation\b", "identification"),
            (r"\bR\s+isk\b", "Risk"),
            (r"\bdefi\s+ned\b", "defined"),
            (r"\bspecifi\s+c\b", "specific"),
            (r"\bbenefi\s+t\b", "benefit"),
            (r"\bC\s+ontrol\b", "Control"),
            (r"\bA\s+ssessment\b", "Assessment"),
            (r"\bP\s+rocess\b", "Process"),
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def create_chunks(self, text: str, chunk_size: int) -> List[str]:
        chunks = [text[i: i + chunk_size]
                  for i in range(0, len(text), chunk_size)]

        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]

        return chunks
