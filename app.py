import streamlit as st
import tempfile
import os
from rag_engine import load_and_split_pdf, build_vectorstore, build_qa_chain, ask_question

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind — RAG Q&A",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4F46E5;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .source-tag {
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🧠 DocuMind</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a PDF. Ask anything. Get answers grounded in your document.</p>', unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── Sidebar — Upload ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF — research paper, report, manual, etc."
    )

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("🔍 Processing your PDF... this takes ~20 seconds"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                chunks = load_and_split_pdf(tmp_path)
                vectorstore = build_vectorstore(chunks)
                st.session_state.qa_chain = build_qa_chain(vectorstore)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
                st.success(f"✅ Ready! Indexed {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.pdf_name:
        st.info(f"📌 Active: **{st.session_state.pdf_name}**")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. PDF is split into chunks\n2. Chunks are embedded via Gemini\n3. Stored in FAISS vector DB\n4. Your question retrieves top chunks\n5. Gemini generates a grounded answer")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ── Main Chat Area ────────────────────────────────────────────────────────────
if not st.session_state.qa_chain:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            cols = st.columns([1, 5])
            with cols[0]:
                st.caption("Sources:")
            with cols[1]:
                source_html = " ".join([
                    f'<span class="source-tag">{s}</span>'
                    for s in message["sources"]
                ])
                st.markdown(source_html, unsafe_allow_html=True)

# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask_question(st.session_state.qa_chain, prompt)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    source_html = " ".join([
                        f'<span class="source-tag">{s}</span>'
                        for s in sources
                    ])
                    st.caption("Sources:")
                    st.markdown(source_html, unsafe_allow_html=True)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })