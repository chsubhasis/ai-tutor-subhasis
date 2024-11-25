import os
from getpass import getpass
import csv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import numpy as np
import gradio
import sqlite3
import PyPDF2
from langchain.prompts import PromptTemplate

hfapi_key = getpass("Enter you HuggingFace access token:")
os.environ["HF_TOKEN"] = hfapi_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key

set_llm_cache(InMemoryCache())

persist_directory = 'docs/chroma/'
pdf_path = 'AIML.pdf'

####################################
def get_documents():
    print("$$$$$ ENTER INTO get_documents $$$$$")
    
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Extract text from all pages
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"

    print("@@@@@@ EXIT FROM get_documents @@@@@")
    return full_text
####################################
def getTexts():
    print("$$$$$ ENTER INTO getDocSplitter $$$$$")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 128
    )
    texts = text_splitter.split_text(get_documents())
    #print("Page content ", texts)
    print("@@@@@@ EXIT FROM getDocSplitter @@@@@")
    return texts
####################################
def getEmbeddings():
    print("$$$$$ ENTER INTO getEmbeddings $$$$$")
    modelPath="mixedbread-ai/mxbai-embed-large-v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device': device}      # cuda/cpu

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    embedding =  HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )
    
    print("@@@@@@ EXIT FROM getEmbeddings @@@@@")
    return embedding
####################################
def getLLM():
    print("$$$$$ ENTER INTO getLLM $$$$$")
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens= 512,
        do_sample= True,
        temperature = 0.7,
        repetition_penalty= 1.2,
        top_k = 10
    )
    print("@@@@@@ EXIT FROM getLLM @@@@@")
    return llm
####################################
def is_chroma_db_present(directory: str):
    """
    Check if the directory exists and contains any files.
    """
    return os.path.exists(directory) and len(os.listdir(directory)) > 0
####################################
def getRetiriver():
    print("$$$$$ ENTER INTO getRetiriver $$$$$")
    if is_chroma_db_present(persist_directory):
        print(f"Chroma vector DB found in '{persist_directory}' and will be loaded.")
        # Load vector store from the local directory
        vectordb = Chroma(
            persist_directory=persist_directory, 
            embedding_function=getEmbeddings(),
            collection_name="ai_tutor")
    else:
        vectordb = Chroma.from_texts(
            collection_name="ai_tutor",
            texts=getTexts(),
            embedding=getEmbeddings(),
            persist_directory=persist_directory, # save the directory
        )
    
    metadata_filter = {
        "result": "llama"  # ChromaDB will perform a substring search
    }
    
    #retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k":5, "filter": metadata_filter})
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k":5})
    print("@@@@@@ EXIT FROM getRetiriver @@@@@")
    return retriever
####################################
def get_rag_response(query):
  print("$$$$$ ENTER INTO get_rag_response $$$$$")  
  qa_chain = RetrievalQA.from_chain_type(
    llm=getLLM(),
    chain_type="stuff",
    retriever=getRetiriver(),
    return_source_documents=True
  )
  
  #RAG Evaluation
  # Sample dataset of questions and expected answers
  dataset = [
    {"question": "Who is the CEO of Meta?", "expected_answer": "Mark Zuckerberg"},
    {"question": "Who is the CEO of Apple?", "expected_answer": "Tiiiiiim Coooooook"},
  ]
  
  hit_rate, mrr = evaluate_rag(qa_chain, dataset)
  print(f"Hit Rate: {hit_rate:.2f}, Mean Reciprocal Rank (MRR): {mrr:.2f}")
  
  # Retrieve context documents
  result = qa_chain({"query": query})

  print("@@@@@@ EXIT FROM get_rag_response @@@@@")
  return result["result"]
####################################
def evaluate_rag(qa, dataset):
    print("$$$$$ ENTER INTO evaluate_rag $$$$$")
    hits = 0
    reciprocal_ranks = []

    for entry in dataset:
        question = entry["question"]
        expected_answer = entry["expected_answer"]
        
        # Get the answer from the RAG system
        response = qa({"query": question}) 
        answer = response["result"] 
        
        # Check if the answer matches the expected answer
        if expected_answer.lower() in answer.lower():
            hits += 1
            reciprocal_ranks.append(1)  # Hit at rank 1
        else:
            reciprocal_ranks.append(0)

    # Calculate Hit Rate and MRR
    hit_rate = hits / len(dataset)
    mrr = np.mean(reciprocal_ranks)

    print("@@@@@@ EXIT FROM evaluate_rag @@@@@")
    return hit_rate, mrr
####################################
def launch_ui():
    print("$$$$$ ENTER INTO launch_ui $$$$$")
    # Input from user
    in_question = gradio.Textbox(lines=10, placeholder=None, value="query", label='Enter your query')

    # Output prediction
    out_response = gradio.Textbox(type="text", label='RAG Response')
    print("out_response ", out_response)
    # Gradio interface to generate UI
    iface = gradio.Interface(fn = get_rag_response,
                            inputs = [in_question],
                            outputs = [out_response],
                            title = "RAG Response",
                            description = "Write the query and get the response from the RAG system",
                            allow_flagging = 'never')

    iface.launch(share = True)

####################################
if __name__ == "__main__":
    launch_ui()