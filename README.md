# 🤖 Internal HR Assistant – FAISS + GPT-3.5 (RAG-based Chatbot)

This project is a Retrieval-Augmented Generation (RAG) powered chatbot that answers HR-related questions using your internal HR policy documents. It uses FAISS for fast semantic retrieval and OpenAI's GPT-3.5 to generate responses in a natural, conversational tone.

It combines:

- FAISS for vector-based search
- OpenAI GPT-3.5 for language generation
- SentenceTransformers for semantic embeddings
- Streamlit for the chat UI Features

### What does it do?

- Accepts questions in plain English (e.g., "How many paid leaves do I get?")
- Retrieves matching HR policy paragraphs from your internal content
- Generates a clear and human-like answer using GPT-3.5
- Only answers from internal context — no hallucinations
- Powered entirely in-memory using FAISS (no external database needed)

### Setup Instructions

#### Step 1: Clone the Repository
    
#### Step 2: Configure Your OpenAI API Key

- Create a .env file in the root directory and add your OpenAI API key:
- OPENAI_API_KEY=your_openai_api_key_here
Do not commit this file to version control. It keeps your key safe.

#### Step 3: Add Billing to OpenAI (Pay-as-you-go)

To use GPT-3.5, you must add a payment method to your OpenAI account:

Go to https://platform.openai.com/account/billing/overview

Enable pay-as-you-go or add a credit card

Alternatively, you can use an open-source model from Hugging Face like google/gemma-2b-it or deepseek-ai/deepseek-llm-1.3b-chat. You'll need to integrate the Hugging Face Inference API or run models locally (not covered in this repo).

#### Step 4: Install Python and Required Libraries

Make sure Python 3.8 to 3.11 is installed.

#### Step 5: Run the Application
- In the terminal, run the command "streamlit run app.py"
- Once it starts, open your browser and go to:http://localhost:8501
- You’ll see a chat interface where you can type questions like:

"How do I apply for leave?"

"What is the review cycle?"

"How do I report harassment anonymously?"

### How It Works

- Your HR policy content is embedded using sentence-transformers/all-mpnet-base-v2, producing 768-dimensional vectors.
- The vector embeddings are stored in a FAISS in-memory index.
- When a user asks a question, it is also embedded into a vector.
- FAISS searches for the most similar policy chunks using L2 (Euclidean) distance.
- The retrieved chunk(s) are passed as context to GPT-3.5.
- GPT 3.5 generates a response strictly based on the provided internal context.

### Key Concepts

- EmbeddingsText is converted into numerical vectors that capture its meaning. Two similar sentences will have similar vector representations.
- VectorsArrays of numbers (e.g., 768 dimensions) that represent sentences in high-dimensional space.
- L2 DistanceA way of measuring how "far apart" two vectors are. Shorter distance = more similar content.
- FAISSAn open-source library developed by Facebook AI Research that performs fast similarity search over dense vectors. Used here to find the most relevant HR policy chunks.
- RAG (Retrieval-Augmented Generation)Combines search (retrieval) and generation. We first find relevant information, then use GPT to respond.

### Example Queries

Try asking questions like:

"How many leaves do I get?"

"Can I carry forward unused leaves?"

"What is the annual performance cycle?"

"How do I raise a complaint about harassment?"

### Improvements You Can Add

Upload and parse HR policies from PDF files using PyMuPDF or pdfplumber

Use Hugging Face models as a drop-in alternative to OpenAI

Add basic authentication to restrict access

Deploy this app to internal servers (Streamlit Cloud or Docker + EC2)
