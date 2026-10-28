import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#Page COnfig

st.set_page_config(page_title="C++ RAG Chatbot")
st.title("C++ RAG Chatbot")
st.write("Ask any questions related to C++ Introduction")

#Load Environment Variables

@st.cache_resource
def load_vectorstore():
    #Load Document
    loader = TextLoader("CPP.txt",encoding="utf-8")
    documents = loader.load()

    #Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents=text_splitter.split_documents(documents)

    #Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    #Vector Storing
    db = FAISS.from_documents(final_documents, embeddings)
    return db

#Load Vector DB
db=load_vectorstore()

#User Input
query = st.text_input("Enter your question about C++")
if query:
    docs=db.similarity_search(query,k=3)
    st.subheader("Retrieved COntext:")
    for i,doc in enumerate(docs):
        st.markdown(f"**Result {i+1}:**")
        st.write(doc.page_content)
