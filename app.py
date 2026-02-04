import chainlit as cl
import torch

# ------------------------------------------------------------------
# IMPORTS (No langchain.chains)
# ------------------------------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Modern LCEL Imports (The Core components)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Use a small model for testing, change this to your heavy model if needed
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

# ------------------------------------------------------------------
# 1. LOAD MODEL
# ------------------------------------------------------------------
def load_llm():
    print(f"Loading Model: {LLM_MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95
    )
    
    return HuggingFacePipeline(pipeline=pipe)

# ------------------------------------------------------------------
# 2. PROCESS PDFS
# ------------------------------------------------------------------
def process_pdfs(files):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = []

    for file in files:
        loader = PyPDFLoader(file.path)
        pages = loader.load()
        splits = text_splitter.split_documents(pages)
        all_splits.extend(splits)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docsearch = FAISS.from_documents(all_splits, embeddings)
    return docsearch

# Helper to format docs into a string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ------------------------------------------------------------------
# 3. CHAINLIT LOGIC
# ------------------------------------------------------------------
@cl.on_chat_start
async def start():
    # 1. Ask for Files
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload your PDF files to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            max_files=5,
            timeout=180
        ).send()

    msg = cl.Message(content=f"Processing {len(files)} files...")
    await msg.send()

    # 2. Process PDFs (Vector DB)
    docsearch = await cl.make_async(process_pdfs)(files)
    retriever = docsearch.as_retriever()

    msg.content = "PDFs processed! Loading AI... (Please wait)"
    await msg.update()

    # 3. Load LLM
    try:
        llm = await cl.make_async(load_llm)()
    except Exception as e:
        await cl.Message(content=f"Error loading model: {e}").send()
        return

    # 4. Build the Chain (Pure LCEL - No Legacy Chains)
    
    # Define the Prompt
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Define the Chain Structure
    # This runs the retriever, then formats docs, then passes to LLM
    rag_chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        .assign(answer=(
            RunnableParallel({
                "context": (lambda x: format_docs(x["context"])),
                "question": (lambda x: x["question"])
            })
            | prompt 
            | llm 
            | StrOutputParser()
        ))
    )

    cl.user_session.set("chain", rag_chain)

    msg.content = "Ready! Ask me anything about your PDFs."
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    if not chain:
        await cl.Message(content="Session expired. Please refresh.").send()
        return

    # Invoke the chain
    res = await cl.make_async(chain.invoke)(message.content)
    
    answer = res["answer"]
    source_documents = res["context"] # The raw docs are preserved here

    # Format sources for UI
    text_elements = []
    if source_documents:
        for i, doc in enumerate(source_documents):
            source_name = f"source_{i}"
            text_elements.append(
                cl.Text(content=doc.page_content, name=source_name, display="side")
            )
        answer += f"\n\n**Sources:** {[f'source_{i}' for i in range(len(source_documents))]}"

    await cl.Message(content=answer, elements=text_elements).send()