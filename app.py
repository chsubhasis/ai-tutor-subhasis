import os
from getpass import getpass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import gradio
import PyPDF2
import json
import re
import time
import threading
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

hfapi_key = getpass("Enter you HuggingFace access token:")
os.environ["HF_TOKEN"] = hfapi_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key

set_llm_cache(InMemoryCache())  # Set cache globally

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
def getTextSplits():
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

    model_kwargs = {
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'stream': True  # Ensure streaming is enabled
    }

    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens= 512,
        do_sample= True,
        temperature = 0.7,
        repetition_penalty= 1.2,
        top_k = 10
        #model_kwargs=model_kwargs  # Pass the model configuration options
    )
    print("@@@@@@ EXIT FROM getLLM @@@@@")
    return llm
####################################
def is_chroma_db_present(directory: str):

    #Check if the directory exists and contains any files.
    return os.path.exists(directory) and len(os.listdir(directory)) > 0
####################################
def getRetiriver(query, metadata_filter:None):
    print("$$$$$ ENTER INTO getRetiriver $$$$$")
    
    # Classify query
    query_type = classify_query(query)
    print("Query classification", query_type)

    k_default = 2
    fetch_k_default = 5
    search_type_default = "mmr"
  
    # Routing logic
    if query_type == 'concept':
        # For conceptual queries, prioritize comprehensive context
        k_default = 5
        fetch_k_default = 10
        search_type_default = "mmr"
    elif query_type == 'example':
        # For example queries, focus on more specific, relevant contexts
        search_type_default = "similarity"
    elif query_type == 'code':
        # For code-related queries, use a more targeted retrieval
        search_type_default = "similarity"

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
            texts=getTextSplits(),
            embedding=getEmbeddings(),
            persist_directory=persist_directory, # save the directory
        )
    
    print("metadata_filter", metadata_filter)
    if(metadata_filter):
        metadata_filter_dict = {
        "result": metadata_filter  # ChromaDB will perform a substring search
        }
        print("@@@@@@ EXIT FROM getRetiriver with metadata_filter @@@@@")

        if search_type_default == "similarity":
            return vectordb.as_retriever(search_type=search_type_default, search_kwargs={"k": k_default, "filter": metadata_filter_dict})
        
        return vectordb.as_retriever(search_type=search_type_default, search_kwargs={"k": k_default, "fetch_k":fetch_k_default, "filter": metadata_filter_dict})

    print("@@@@@@ EXIT FROM getRetiriver without metadata_filter @@@@@")
    if search_type_default == "similarity":
        return vectordb.as_retriever(search_type=search_type_default, search_kwargs={"k": k_default})
    
    return vectordb.as_retriever(search_type=search_type_default, search_kwargs={"k": k_default, "fetch_k":fetch_k_default})
####################################
def classify_query(query):
    """
    Classify the type of query to determine routing strategy.
    
    Query Types:
    - 'concept': Theoretical or conceptual questions
    - 'example': Requests for practical examples
    - 'code': Coding or implementation-related queries
    - 'general': Default catch-all category
    """
    query = query.lower()
    
    # Concept detection patterns
    concept_patterns = [
        r'what is',
        r'define',
        r'explain',
        r'describe',
        r'theory of',
        r'concept of'
    ]
    
    # Example detection patterns
    example_patterns = [
        r'give an example',
        r'show me an example',
        r'demonstrate',
        r'illustrate'
    ]
    
    # Code-related detection patterns
    code_patterns = [
        r'how to implement',
        r'code for',
        r'python code',
        r'algorithm implementation',
        r'write a program'
    ]
    
    # Check patterns
    for pattern in concept_patterns:
        if re.search(pattern, query):
            return 'concept'
    
    for pattern in example_patterns:
        if re.search(pattern, query):
            return 'example'
    
    for pattern in code_patterns:
        if re.search(pattern, query):
            return 'code'
    
    return 'general'
####################################
def get_rag_response(query, metadata_filter=None):
    print("$$$$$ ENTER INTO get_rag_response $$$$$")
  
    # Create the retriever
    retriever = getRetiriver(query, metadata_filter)

    # Get the LLM
    llm = getLLM()

    # Create a prompt template
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    
    prompt = PromptTemplate.from_template(template)

    # Function to prepare input for the chain
    def prepare_inputs(inputs):
        retrieved_docs = retriever.invoke(inputs["question"])
        context = format_docs(retrieved_docs)
        return {
            "context": context,
            "question": inputs["question"]
        }

    # Construct the RAG chain with streaming
    rag_chain = (
        RunnablePassthrough()
        | RunnableLambda(prepare_inputs)
        | prompt
        | llm
        | StrOutputParser()
    )

    # Stream the response
    full_response = ""
    for chunk in rag_chain.stream({"question": query}):
        full_response += chunk
        # Add a small delay to create a streaming effect
        time.sleep(0.05)  # 50 milliseconds between chunk updates
        yield full_response

####################################
# Utility function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
####################################
# Input from user
in_question = gradio.Textbox(lines=10, placeholder=None, value="What are Artificial Intelligence and Machine Learning?", label='Ask a question to your AI Tutor')

# Optional metadata filter input
in_metadata_filter = gradio.Textbox(lines=2, placeholder=None, label='Optionally add a filter word')

# Output prediction
out_response = gradio.Textbox(label='Response', interactive=False, show_copy_button=True)

# Gradio interface to generate UI
iface = gradio.Interface(
    fn = get_rag_response,
    inputs=[in_question, in_metadata_filter],
    outputs=out_response,
    title="Your AI Tutor",
    description="Ask a question, optionally add metadata filters.",
    allow_flagging='never',
    stream_every=0.5
    )

iface.launch(share = True)