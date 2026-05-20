import chromadb as db #vector database
from sentence_transformers import SentenceTransformer as st #embeddings
import os #file path handling
from dotenv import load_dotenv as de #.env secrets
from docx import Document as doc
from unstructured.partition.auto import partition   # extracts text from ANY file
from langchain_text_splitters import RecursiveCharacterTextSplitter  # chunking

z=os.getcwd()
#print("Current folder:", z)
path=os.listdir(z)
#print("current list: ",path) 
path=os.path.join(z,"data")
#print("current list: ",path) 
filepath=os.listdir(path)
#print("All files are: ", filepath)
f=[]

for files in filepath:
    print("Name of file: ", files)
    ft=os.path.join(path,files)
    element=partition(ft)
    finaltext=('\n'.join([str(e) for e in element]))
    f.append({"source": files, "text": finaltext})
'''
for item in f:
    print("SOURCE:", item["source"])
    print("PREVIEW:", item["text"][:100])
    print("---")
'''

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks=[]
for item in f:
    chunk=splitter.split_text(item["text"])
    for index,sentence in enumerate(chunk):
        chunks.append({"source": item["source"],"text": sentence, "chunk_id": f"{item['source']}_{index}"
})

#print(f"Total chunks: {len(chunks)}")
#print(chunks[0])
model=st('all-MiniLM-L6-v2')

for content in chunks:
    embedding=model.encode(content["text"])
    embedding_list=embedding.tolist()
    content["embedding"]= embedding_list
print(len(chunks[0]["embedding"]))  # should print 384    

client = db.PersistentClient(path="./chroma_db")
collection=client.get_or_create_collection(
   name="xmlgenerator", metadata={"hnsw:space": "cosine"})


collection.upsert(
    ids=[x["chunk_id"] for x in chunks],
    embeddings=[x["embedding"] for x in chunks],
    documents=[x["text"] for x in chunks],
    metadatas=[{"source": x["source"]} for x in chunks])   