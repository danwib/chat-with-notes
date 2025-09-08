import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory  # updated import


DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db")


def run_chat():
    load_dotenv()  # loads OPENAI_API_KEY if present

    # --- Vector store + retriever ---
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # --- LLM ---
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # --- Conversational Retrieval Chain (no legacy Memory object) ---
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
    )

    # --- New-style history ---
    history = ChatMessageHistory()

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
            # Pass chat history explicitly
            result = qa.invoke(
                {
                    "question": query,
                    "chat_history": history.messages,
                }
            )
            answer = result["answer"]

            # Update history
            history.add_user_message(query)
            history.add_ai_message(answer)

        except Exception as e:
            print(f"[Error] {e}")
            continue

        print("Bot:", answer)


if __name__ == "__main__":
    run_chat()
