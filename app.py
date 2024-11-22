import os
from getpass import getpass
import csv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import numpy as np
import gradio

hfapi_key = getpass("Enter you HuggingFace access token:")
os.environ["HF_TOKEN"] = hfapi_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key

set_llm_cache(InMemoryCache())

persist_directory = 'docs/chroma/'

####################################
def load_file_as_JSON():
    rows = []
    with open("mini-llama-articles.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue
                # Skip header row
            rows.append(row)
    return rows
####################################
def get_documents():
    documents = [
        Document(
            page_content=row[1], metadata={"title": row[0], "url": row[2], "source_name": row[3]}
        )
        for row in load_file_as_JSON()
    ]
    print("documents lenght is ", len(documents))
    print("first entry from documents ", documents[0])
    print("document metadata ", documents[0].metadata)
    return documents
####################################
def getDocSplitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 128
    )
    splits = text_splitter.split_documents(get_documents)
    print("Split length ", len(splits))
    print("Page content ", splits[0].page_content)
    return splits
####################################
def getEmbeddings():
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
    
    print("Embedding ", embedding)
    return embedding
####################################
def getLLM():
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        #repo_id="chsubhasis/ai-tutor-towardsai",
        task="text-generation",
        max_new_tokens = 512,   
        top_k = 10,
        temperature = 0.1,
        repetition_penalty = 1.03,
    )
    print("llm ", llm)
    print("Who is the CEO of Apple? ", llm.invoke("Who is the CEO of Apple?")) #test
    return llm
####################################
def getRetiriver():
    vectordb = Chroma.from_documents(
        documents=getDocSplitter(), # splits we created earlier
        embedding=getEmbeddings(),
        persist_directory=persist_directory, # save the directory
    )
    print("vectordb collection count ", vectordb._collection.count())
    
    docs = vectordb.search("What is Artificial Intelligence", search_type="mmr", k=5)
    for i in range(len(docs)):
     print(docs[i].page_content)
    
    metadata_filter = {
        "result": "llama"  # ChromaDB will perform a substring search
    }
    
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k":5, "filter": metadata_filter})
    print("retriever ", retriever)
    return retriever
####################################
def get_rag_response(query):
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
  
  result = qa_chain({"query": query})
  print("Result ",result)
  return result["result"]
####################################
def evaluate_rag(qa, dataset):
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

    return hit_rate, mrr
####################################
def launch_ui():
    # Input from user
    in_question = gradio.Textbox(lines=10, placeholder=None, value="query", label='Enter your query')

    # Output prediction
    out_response = gradio.Textbox(type="text", label='RAG Response')

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