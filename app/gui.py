import faiss
import pickle
from sentence_transformers import SentenceTransformer
#import openai
import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

client = OpenAI()

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Load the FAISS Index and Metadata
def load_vector_db(index_path: str, metadata_path: str):
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            passages = pickle.load(f)
        return index, passages
    except Exception as e:
        logging.error(f"Error loading vector DB: {e}")
        return None, []

def retrieve_similar_passages(query: str, index, passages, model, top_k=3, max_tokens=3000):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    selected_passages = []
    total_tokens = 0

    for i in indices[0]:
        passage = passages[i]
        token_count = len(passage.split())

        if total_tokens + token_count <= max_tokens:
            selected_passages.append(passage)
            total_tokens += token_count
        else:
            break

    return selected_passages

def generate_answer_with_openai(query: str, context: str, max_tokens=3000):
    context_words = context.split()
    if len(context_words) > max_tokens:
        context = ' '.join(context_words[:max_tokens])

    prompt = f"Kontext: {context}\n\nFrage: {query}\nAntwort:"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Assistent. Beantworte die Fragen auf Deutsch."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return "Es gab ein Problem bei der Generierung der Antwort."

# RAG Pipeline Function
def ask_question(query: str):
    index_path = "./vector_db/activities_index.faiss"
    metadata_path = "./vector_db/activities_passages.pkl"

    index, passages = load_vector_db(index_path, metadata_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    similar_passages = retrieve_similar_passages(query, index, passages, model)

    if similar_passages:
        context = " ".join(similar_passages)
        answer = generate_answer_with_openai(query, context)
    else:
        answer = generate_answer_with_openai(query, "Keine relevanten Ergebnisse gefunden. Bitte generiere eine Antwort basierend auf der Frage.")
    
    return answer

# Streamlit GUI with conversation buffer
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFAFA; /* Light background */
    }
    .chat-box {
        border: 1px solid #CCC;
        padding: 10px;
        margin: 10px 0;
        border-radius: 10px;
    }
    .chat-query {
        color: red;
        font-weight: bold;
    }
    .chat-response {
        color: green;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("app/logo.png", width=100)
st.markdown("<h2 class='centered-header'>👦👧 Kinder-Aktivitätenerfinder! 👧👦</h2>", unsafe_allow_html=True)

# User input at the bottom
query = st.text_input("🤖 Stellen Sie eine Frage zu Aktivitätsideen...", "")

# Button press logic
if st.button("Suchen"):
    if query:
        answer = ask_question(query)

        # Update session state immediately with the new query and answer
        st.session_state.conversation.append({"query": query, "answer": answer})

# Conversation display first
for entry in reversed(st.session_state.conversation):  # Reverse to show newest first
    st.markdown(f"<div class='chat-box'><span class='chat-query'>🧑 </span> {entry['query']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-box'><span class='chat-response'>🤖 </span> {entry['answer']}</div>", unsafe_allow_html=True)
