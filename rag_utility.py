import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import torch
import sentence_transformers



# Load enviromemt variables from .env file
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

#Load the embedding model
embedding = HuggingFaceEmbeddings()

# Load the Llama-3.3-70b model from Groq
llm =ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature=0
)

def  process_document_to_chroma_db(file_name):
    # Load the PDF document using UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()
    #Split the text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 200
    )
    texts = text_splitter.split_documents(documents)
    #Store the document chunks in a Chroma vector database
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0 

def answer_question(user_question):
    # Load the presistent Chroma vector database 
    vector_db = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function = embedding
    )
    # Create a retriver for document search
    retriever = vector_db.as_retriever()

    #Create a RetrievelQA.from_chain_type
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm ,
        chain_type="stuff",
        retriever=retriever,
    )
    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]


    return answer
