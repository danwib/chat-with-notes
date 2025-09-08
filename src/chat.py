import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db")

def run_chat():
    load_dotenv()  # load OPENAI_API_KEY from .env if present

    # Vector store + retriever
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # swap model if you like

    # Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Conversational Retrieval QA
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    print("Chat with your notes! (type 'exit' to quit)")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            print("Goodbye!")
            break

        try:
            answer = qa.run(query)
        except Exception as e:
            print(f"[Error] {e}")
            continue

        print("Bot:", answer)

if __name__ == "__main__":
    run_chat()
