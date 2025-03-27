import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# 1. Load your OpenAI API key securely from the .env file
# -------------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------------------
# 2. Load a sentence embedding model (768-dimensional output)
# This model converts text into vector representations
# -------------------------------------------------------------------
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# -------------------------------------------------------------------
# 3. Define your HR policy documents
# Each document will be embedded and stored in a vector database
# -------------------------------------------------------------------
hr_docs = [
    "Employees are entitled to 24 paid leaves annually. This includes sick leaves, casual leaves, and earned leaves.",
    "To apply for leave, employees must use the internal HR portal and select the appropriate leave type. Leave requests must be approved by their manager.",
    "Unused leaves can be carried forward to the next calendar year, up to a maximum of 15 days.",
    "The annual performance review cycle begins in March and ends in May. Each employee will be reviewed by their direct manager.",
    "Performance is evaluated based on individual KPIs, team contribution, and feedback from peers.",
    "The performance rating impacts annual bonuses and promotions, and all reviews are documented in the HRMS system.",
    "To file a complaint regarding harassment, discrimination, or any people-related issues, employees must contact the HR team via hr-complaints@company.com.",
    "Complaints can also be filed anonymously using the internal ethics hotline.",
    "All complaints are treated confidentially and are handled as per the company’s grievance redressal policy."
]

# -------------------------------------------------------------------
# 4. Embed and index the HR policy vectors into FAISS (one-time only)
# Streamlit reruns the script often, so we use session_state to avoid re-indexing
# -------------------------------------------------------------------
if "faiss_index" not in st.session_state:
    # Convert each policy into a dense 768-dimensional embedding
    embeddings = embedder.encode(hr_docs)  # shape: (num_docs, 768)

    # Get the dimensionality of each vector (should be 768)
    dim = embeddings.shape[1]

    # Create a FAISS index using L2 (Euclidean) distance
    index = faiss.IndexFlatL2(dim)

    # Add all the document embeddings into the FAISS index
    index.add(np.array(embeddings))

    # Store everything in session state so it persists across reruns
    st.session_state.faiss_index = index
    st.session_state.hr_docs = hr_docs
    st.session_state.embeddings = embeddings

# -------------------------------------------------------------------
# 5. Check if the user just said "hi" or greeted us
# We use this to avoid calling the LLM unnecessarily
# -------------------------------------------------------------------
def is_small_talk(text):
    return text.strip().lower() in ["hi", "hello", "hey", "good morning", "good evening"]

# -------------------------------------------------------------------
# 6. Retrieve the most relevant HR policy chunks based on user query
# Uses FAISS to search for the nearest vector in semantic space
# -------------------------------------------------------------------
def retrieve_context(question, k=3, max_l2_distance=1.5):
    query_embedding = embedder.encode([question])  # vector for user query

    # D = distances, I = indices (top k closest vectors)
    D, I = st.session_state.faiss_index.search(query_embedding, k)
    distances = D[0]
    indices = I[0]

    # Optional debug to monitor retrieval quality
    print("Top distance score:", distances[0])

    # If top result is not close enough, return None
    if distances[0] > max_l2_distance:
        return None

    # Otherwise return the top k most relevant policy texts
    return [st.session_state.hr_docs[i] for i in indices]

# -------------------------------------------------------------------
# 7. Query OpenAI GPT-3.5 to generate a final answer from the context
# We pass both the HR context and the user question to GPT
# -------------------------------------------------------------------
def query_openai(user_input, context):
    messages = [
        {"role": "system", "content": "You are an internal HR assistant. Use the given company policy context to answer accurately and concisely."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# -------------------------------------------------------------------
# 8. Streamlit UI setup
# -------------------------------------------------------------------
st.set_page_config(page_title="HR Assistant", layout="wide")
st.title("🤖 Internal HR Assistant (FAISS + GPT-3.5)")

# Store full conversation history across Streamlit reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages (if any)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------------------------
# 9. Accept user input and handle the query
# -------------------------------------------------------------------
user_input = st.chat_input("Ask me anything about HR policies...")

if user_input:
    # Show user’s message in chat
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Check for greetings
    if is_small_talk(user_input):
        reply = "Hello! How can I assist you today with HR-related queries?"

    else:
        # Step 1: Retrieve most relevant HR policy content
        chunks = retrieve_context(user_input)

        # Step 2: Handle fallback if nothing relevant is found
        if not chunks:
            reply = "I'm sorry, I couldn't find any information related to your query in our current HR policies."

        # Step 3: Generate GPT response using retrieved context
        else:
            context = "\n".join(chunks)
            with st.spinner("Thinking..."):
                reply = query_openai(user_input, context)

    # Show assistant reply
    st.chat_message("assistant").markdown(reply)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
