# Chat with Your Notes (LangChain Starter)

A tiny project to learn **LangChain** by building a **chatbot over local notes** (txt/PDF).  
It covers: loading docs, chunking, embeddings, a vector store (Chroma), retrieval, LLM calls, and basic conversation memory.

---

## Quick Start

### 1) Create environment & install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Set your API key
Create a `.env` file (or set an env var) with:
```
OPENAI_API_KEY=your_key_here
```
Alternatively:
```bash
export OPENAI_API_KEY="your_key_here"  # Windows PowerShell: $Env:OPENAI_API_KEY="your_key_here"
```

### 3) Add notes
Drop `.txt` or `.pdf` files into the `data/` folder.

### 4) Build the vector DB
```bash
python src/ingest.py
```

### 5) Chat!
```bash
python src/chat.py
```

You should see:
```
Chat with your notes! (type 'exit' to quit)
You:
```

---

## Features Learned
- **Document loading** for txt/PDF
- **Chunking** with `RecursiveCharacterTextSplitter`
- **Embeddings** via `OpenAIEmbeddings`
- **Vector DB** using `Chroma` with persistence (`./db`)
- **Retrieval + LLM** via `ConversationalRetrievalChain`
- **Conversation memory** with `ConversationBufferMemory`

---

## Tech choices & notes
- Uses the **modern LangChain import split** (`langchain`, `langchain_community`, `langchain_openai`).
- Chroma persists locally to `./db` (safe to delete and re‑run `ingest.py`).
- Defaults to `gpt-4o-mini` (cheap/fast) — tweak in `src/chat.py`.

---

## Common tweaks
- Change chunk sizes/overlap in `ingest.py` to suit your docs.
- Swap `ConversationBufferMemory` for summary or token‑aware memory.
- Add a minimal web UI (e.g., Streamlit/FastAPI) later.

---

## Repo layout
```
chat-with-notes/
├── data/                  # Place your notes here (.txt/.pdf)
├── db/                    # Chroma persistence (auto-created)
├── src/
│   ├── ingest.py          # Build embeddings + vector store
│   └── chat.py            # CLI chat over your notes
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Troubleshooting
- **No docs found**: Ensure files exist in `data/` and re-run `ingest.py`.
- **Embedding/LLM errors**: Check that `OPENAI_API_KEY` is set.
- **Import errors**: `pip install -r requirements.txt` again; versions pinned.

---

## 📝 License
MIT — use freely.
