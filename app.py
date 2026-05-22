import streamlit as st
import tempfile
from RAG_pipeline import ask
import os
from unstructured.partition.auto import partition

st.title("Survey Co-Pilot")
st.caption("Let's build you a legit survey dummy ;)")
if "messages" not in st.session_state:
    st.session_state.messages=[]

uploaded_file=st.file_uploader(
    "Upload your draft",
    type=["docx","pdf"])    

draft_text=""
if uploaded_file:
    with tempfile.NamedTemporaryFile (delete=False,suffix=f"_{uploaded_file.name}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path=tmp.name
    elements=partition(tmp_path)
    os.unlink(tmp_path)
    draft_text= "\n".join([str(e) for e in elements])
    st.success("Draft uploaded successfully ✅")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


prompt=st.chat_input("What's up?")

if draft_text and prompt:
    user_input = f"DRAFT:\n{draft_text}\n\nINSTRUCTIONS:\n{prompt}"
elif draft_text:
    user_input = draft_text
else:
    user_input = prompt

if user_input:
    # Save and show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get and show RAG answer
    with st.chat_message("assistant"):
        with st.spinner("Generating XML..."):
            answer = ask(user_input)
        st.write(answer)
    
    # Save assistant answer
    st.session_state.messages.append({"role": "assistant", "content": answer})    