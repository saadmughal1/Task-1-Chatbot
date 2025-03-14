import streamlit as st
import os
import time
import pickle
import pandas as pd
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

FAISS_INDEX_PATH = "faiss_index"

# Function to load FAISS only once
def store_in_faiss(csv_file, faiss_index_path=FAISS_INDEX_PATH):
    if "faiss_loaded" in st.session_state:
        return

    with st.spinner("🔄 Loading medical knowledge base... Please wait."):

        df = pd.read_csv(csv_file)

        text_list = df["text"].tolist()
        names_list = df["name"].tolist()

        vector_store = FAISS.from_texts(text_list, embeddings)
        vector_store.save_local(faiss_index_path)

        with open(f"{faiss_index_path}/metadata.pkl", "wb") as f:
            pickle.dump(names_list, f)

        time.sleep(1)
        st.success("✅ Medical knowledge base loaded successfully!")
        st.session_state.faiss_loaded = True

# Function to get the conversational chain using LLMChain
def get_conversational_chain():
    prompt_template = """
    You are a helpful and knowledgeable AI medical assistant. Maintain a natural and conversational tone. 
    Use past conversation context to continue discussions smoothly. If the answer is not available in the provided context, 
    say: "I'm sorry, but I don't have enough medical information on that topic.

    - If the user uses pronouns like "it" or "this," assume they refer to the most recent topic.
    - Do NOT explicitly say "Given our previous conversation." Instead, naturally continue.
    - Keep responses clear, concise, and engaging.

    Chat History:
    {chat_history}

    Context:
    {context}

    User Query:
    {question}

    AI Response:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)
    return LLMChain(prompt=prompt, llm=model)

# Function to handle user input & generate response
def user_input(user_question):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local(FAISS_INDEX_PATH, embedding, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)  # Retrieve more relevant documents
    
    context_text = "\n".join([doc.page_content for doc in docs])

    chat_history = "\n".join([msg["role"] + ": " + msg["content"] for msg in st.session_state.chat_history])

    chain = get_conversational_chain()
    response = chain.run({"chat_history": chat_history, "context": context_text, "question": user_question})

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    return response

# Streamlit UI
def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon="⚕️", layout="wide")
    st.title("⚕️ AI Medical Chatbot")

    store_in_faiss("medical-dataset.csv")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_message = st.chat_input("Ask me anything about symptoms, diseases, or treatments...")

    if user_message:
        with st.spinner("Thinking... 🤖"):
            ai_response = user_input(user_message)

    for message in st.session_state.chat_history:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message["content"])

if __name__ == "__main__":
    main()
