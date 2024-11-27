## Overview

This repository contains the code of the final "Part 4; Building Your Own advanced LLM + RAG Project to receive certification" lesson of the "From Beginner to Advanced LLM Developer" course.

## Notes
- AI_Tutor_FT.ipynb is only to fine tune the model and save in Hugging Face space. This will not be required to exeute to launch the app. 
- AI_Tutor_RAG.ipynb is for RAG implementation. This is modularized and moved to app.py.
- This project used below:
    - Data collection and curation process leverages PDFs. (Ref: AIML.pdf)
    - Prompt caching done using InMemoryCache and set_llm_cache from langchain_core.caches and langchain_core.globals respectively. Search using the same prompt and see the performance improvement. set_llm_cache(InMemoryCache()) does the work. Refer: https://python.langchain.com/docs/how_to/llm_caching/
    - Metadata filtering done based on dynamic data. Retriever is updated accordingly.
    - Query routing: Based on the query type, search_type is altered in run time ("mmr" or "similarity"). Try with prompts like "Demonstrate Artificial Intelligence and Machine Learning." and "What are Artificial Intelligence and Machine Learning?"
    - Query pipeline includes function calling.
    - 

## Setup

1. Create a local virtual environment, for example using the `venv` module. Then, activate it.

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the dependencies. (Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/. Otherwise langchain_chroma installtion will fail.)

```bash
pip install -r requirements.txt
```

4. Launch the Gradio app.

```bash
python app.py
```