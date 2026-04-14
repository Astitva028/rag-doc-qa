import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def load_and_split_pdf(pdf_path: str) -> list:
    """Load a PDF and split it into overlapping chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)
    return chunks


def build_vectorstore(chunks: list) -> FAISS:
    """Embed chunks using Gemini embeddings and store in FAISS."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def build_qa_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    """Build a conversational RAG chain with memory."""
    llm = ChatGoogleGenerativeAI(
       model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return qa_chain


def ask_question(chain: ConversationalRetrievalChain, question: str) -> dict:
    """Ask a question and return the answer + source pages."""
    result = chain({"question": question})

    sources = list(set([
        f"Page {doc.metadata.get('page', 'N/A') + 1}"
        for doc in result.get("source_documents", [])
    ]))

    return {
        "answer": result["answer"],
        "sources": sources
    }