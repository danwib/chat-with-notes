from pathlib import Path
from typing import List
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DB_DIR = Path(__file__).resolve().parents[1] / "db"

def load_documents(paths: List[Path]):
    docs = []
    for p in paths:
        if p.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
        elif p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
    return docs

def ingest():
    load_dotenv()  # load OPENAI_API_KEY if in .env
    files = [p for p in DATA_DIR.glob("*") if p.suffix.lower() in {".txt", ".pdf"}]
    if not files:
        raise SystemExit(f"No .txt or .pdf files found in {DATA_DIR}. Add notes and try again.")

    print(f"Loading {len(files)} file(s) from {DATA_DIR}...")
    documents = load_documents(files)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s).")

    # Build embeddings + Chroma DB
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=str(DB_DIR))
    db.persist()
    print(f"Persisted vector DB to {DB_DIR}")

if __name__ == "__main__":
    ingest()
