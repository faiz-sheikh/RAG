
import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


st.title("College Knowledge Assistant")


# Upload PDF and TXT files
uploaded_files = st.file_uploader(
    "Upload college documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload at least five PDF or TXT documents.")
    st.stop()

if len(uploaded_files) < 5:
    st.warning("Please upload at least five documents.")
    st.stop()


# Load documents and create the vector database

documents = []

for file in uploaded_files:
    with open(file.name, "wb") as f:
        f.write(file.getvalue())

    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(file.name)
    else:
        loader = TextLoader(file.name, encoding="utf-8")

    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = file.name

    documents.extend(docs)

    os.remove(file.name)

# Split documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks = splitter.split_documents(documents)


# Convert chunks into embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store embeddings in FAISS
db = FAISS.from_documents(chunks, embeddings)


# Configure the local language model
llm = ChatOllama(model="gemma2:2b")

prompt = ChatPromptTemplate.from_template("""
You are a college assistant. Answer using only the context below.

If the answer is missing from the context, say:
"The available college documents do not contain enough information
to answer this question."

If the sources contradict each other, explain the conflict and
identify the sources. Do not invent facts or citations.

Context:
{context}

Question:
{question}

Answer with the relevant source names.
""")

chain = prompt | llm | StrOutputParser()


# Ask a question
query = st.text_input("Ask a question about college documents:")

if query:
    # Retrieve the three most similar chunks
    docs = db.similarity_search(query, k=3)

    # Prepare context for the language model
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', 'N/A')}\n"
        f"{doc.page_content}"
        for doc in docs
    )

    # Generate an answer using the retrieved documents
    answer = chain.invoke({
        "context": context,
        "question": query
    })

    st.subheader("Answer")
    st.write(answer)

    # Display retrieved source passages
    st.subheader("Supporting Sources")

    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        page = page + 1 if page is not None else "N/A"

        with st.expander(
            f"Source {i}: {doc.metadata.get('source', 'Unknown')} "
            f"(Page {page})"
        ):
            st.write(doc.page_content)
