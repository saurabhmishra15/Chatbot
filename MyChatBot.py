import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.header("BunnyyBot")

if not GOOGLE_API_KEY:
    st.error("Missing API key. Set GOOGLE_API_KEY in your .env file.")
    st.stop()

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader(
        "Upload notes PDF and start asking questions",
        type="pdf"
    )


@st.cache_resource(show_spinner="Reading and indexing your notes...")
def build_vector_store(file_bytes):

    my_pdf = PdfReader(file_bytes)

    text = ""

    for page in my_pdf.pages:
        text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )

    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    return FAISS.from_texts(chunks, embeddings)


if file is not None:

    vector_store = build_vector_store(file)

    user_query = st.text_input("Type your query here")

    if user_query:

        matching_chunks = vector_store.similarity_search(user_query)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            max_tokens=300
        )

        customized_prompt = ChatPromptTemplate.from_template(
            """You are my assistant tutor.
Answer the question based on the following context.

If you did not get the context, simply say "I don't know."

Context:
{context}

Question:
{input}
"""
        )

        chain = create_stuff_documents_chain(
            llm,
            customized_prompt
        )

        output = chain.invoke({
            "input": user_query,
            "context": matching_chunks
        })

        st.write(output)