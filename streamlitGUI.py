import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Pinecone API setup
PINECONE_API_KEY = "1446a3fc-441d-4dc6-befc-aa31b939bfdf"
PINECONE_API_ENV = "starter"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"
index = pc.Index(index_name)

# Download and initialize the Hugging Face embeddings model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()

# Streamlit app title and description
st.title("Dr.Razin's Medical Chatbot")

# Query Section
query = st.text_input("Enter your query:")

if query:
    # Generate embedding for the query
    query_embedding = embeddings.embed_query(query)

    # Perform similarity search
    search_results = index.query(
        namespace="ns1",
        vector=query_embedding,
        top_k=1,
        include_values=True,
        include_metadata=True
    )

    # Display the top result
    if search_results['matches']:
        answer = search_results['matches'][0]['metadata']['text']
        st.write(f"Answer: {answer}")
    else:
        st.write("No relevant information found.")
