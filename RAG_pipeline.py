from groq import Groq
from dotenv import load_dotenv
import chromadb as db
from sentence_transformers import SentenceTransformer
import os

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

db_client = db.PersistentClient(path="./chroma_db")
collection=db_client.get_or_create_collection(
   name="xmlgenerator", metadata={"hnsw:space": "cosine"})

model=SentenceTransformer("all-MiniLM-L6-v2")

def build_prompt(question_text, chunks):
    messages = [
    {
        "role": "system",
        "content": """
        You are an expert Decipher survey XML programmer with 5+ years of experience.
        Your job is to generate valid Decipher XML based on the user's survey draft,
        using only the retrieved examples from our CLB and XML reference files.

        RULES FOR XML GENERATION:
        - Question number goes in the label attribute
        - Question text goes inside <title></title> tags
        - Answer options go inside <row></row> tags
        - Single select question → use <radio> tag
        - Multi select question → use <checkbox> tag  
        - Numeric input question → use <number> tag
        - Open text / typed answer question → use <textarea> tag
        - Grid with rows and columns → use <radio> with <row> and <col> tags

        OUTPUT RULES:
        - Always generate complete, valid XML
        - Follow the exact patterns from retrieved reference documents
        - Mention which reference file you used at the end
        - If no relevant reference found in retrieved context, respond exactly with:
          'I don't have a reference for this question type in my knowledge base.'
        - Never generate XML from your own knowledge — only from retrieved examples"""
    },
    {
        "role": "user",
        "content": f"CONTEXT:\n{chunks}\nQUESTION: {question_text}"
    }
]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
)

    return response.choices[0].message.content

def ask(question_text):
    embed=model.encode(question_text).tolist()
    results=collection.query(
        query_embeddings=[embed],
        n_results=3
    )
    texts=results["documents"][0]
    sources=results["metadatas"][0]
    context=""
    for text, meta in zip(texts,sources):
        context += f"--- {meta['source']} ---\n"
        context += f"{text}\n\n"
    
    return build_prompt(question_text, context)


ques=input("Please enter your query")
ques=ques.lower()
print(ask(ques))
