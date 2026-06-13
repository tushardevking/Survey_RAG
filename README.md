# Survey RAG — Intelligent Survey Assistant

A RAG (Retrieval Augmented Generation) pipeline built for 
Decipher survey programming. Ask questions in plain English 
and get answers based on your actual XML files and survey templates.

## What This Does
- Ingests Decipher XML files and survey templates
- Converts them into searchable embeddings using ChromaDB
- Answers survey programming questions based on YOUR actual code
- Returns accurate, grounded answers — not generic AI responses

## Tech Stack
- Python
- Groq API (LLaMA 3.3 70B)
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- LangChain (orchestration)

