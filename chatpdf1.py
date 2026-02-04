import streamlit as st
import os
import torch
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------- CONFIGURATION ----------------
load_dotenv()

# Check for token
hf_token = os.getenv("HF_TOKEN")

# ---------------- PDF TEXT EXTRACTION ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# ---------------- TEXT CHUNKING ----------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# ---------------- CREATE VECTOR STORE ----------------
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ---------------- LOAD LLaMA-2 LLM ----------------
@st.cache_resource
def load_llm():
    # If Llama-2 crashes your RAM, try swapping this to: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    # We removed BitsAndBytesConfig because it breaks on Windows.
    # We use torch.float16 to save memory compared to full precision.
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto", 
            torch_dtype=torch.float16, # Helps reduce memory usage on Windows
            token=hf_token
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
        repetition_penalty=1.1
    )
    
    return HuggingFacePipeline(pipeline=pipe)

# ---------------- ASK QUESTION ----------------
def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.warning("Please upload and process your PDFs first.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = db.similarity_search(user_question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""[INST] <<SYS>>
You are a helpful assistant. Answer the question based ONLY on the context provided below. 
If the answer is not in the context, simply state "I cannot find the answer in the provided document."
<</SYS>>

Context:
{context}

Question:
{user_question} [/INST]
"""

    llm = load_llm()
    response = llm.invoke(prompt)
    
    st.write("### Answer:")
    st.write(response)

# ---------------- STREAMLIT UI ----------------
def main():
    st.set_page_config(page_title="Chat PDF with LLaMA-2", layout="wide")
    st.header("Chat with PDF using LLaMA-2 7B 🦙")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("Done!")

    user_question = st.text_input("Ask a question from the PDF files")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()