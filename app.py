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
from transformers import pipeline


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
def getStream(query, metadata_filter=None):
    """
    Stream the RAG response using a generator function.
    
    Args:
        query (str): User's input query
        metadata_filter (str, optional): Metadata filter for retrieval
    
    Yields:
        str: Partial response chunks for streaming
    """
    # Retrieve context
    retriever = getRetiriver(query, metadata_filter)
    documents = retriever.get_relevant_documents(query)
    
    # Prepare context for generation
    context = " ".join([doc.page_content for doc in documents])
    
    # Use HuggingFace text generation pipeline with streaming
    generator = pipeline(
        "text-generation", 
        model="HuggingFaceH4/zephyr-7b-beta", 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Construct prompt with retrieval context
    prompt = f"""
    Context: {context}
    
    Query: {query}
    
    Based on the context, provide a detailed and accurate response to the query:
    """
    
    # Generate response with streaming
    response_chunks = generator(
        prompt, 
        max_new_tokens=512, 
        do_sample=True, 
        temperature=0.7,
        num_return_sequences=1,
        stream=True
    )
    
    full_response = ""
    for chunk in response_chunks:
        # Extract generated text
        generated_text = chunk[0]['generated_text']
        
        # Remove prompt from generated text
        response = generated_text.replace(prompt, '').strip()
        
        # Yield progressive chunks
        if response and response != full_response:
            yield response
            full_response = response
####################################
def launch_ui():
    # Input from user
    in_question = gradio.Textbox(lines=10, placeholder=None, value="query", label='Ask a question to you AI Tutor')
    
    # Optional metadata filter input
    in_metadata_filter = gradio.Textbox(lines=2, placeholder=None, label='Optionally add a filter word')
    
    # Streaming output
    out_response = gradio.Textbox(label='Response', interactive=False)

    out_image = gradio.Image()

    # Gradio interface to generate UI
    iface = gradio.Interface(
        fn = getStream,
        inputs=[in_question, in_metadata_filter],
        outputs=[out_response, out_image],
        title="Your AI Tutor",
        description="Ask a question with optional metadata filters, get streaming responses.",
        allow_flagging='never'
    )

    iface.launch(share = True)

####################################
if __name__ == "__main__":
    launch_ui()