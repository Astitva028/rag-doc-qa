# 🧠 DocuMind — RAG Document Q&A Chatbot

A conversational AI chatbot that answers questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG).

## 🛠️ Tech Stack
- **LangChain** — RAG orchestration & conversational memory
- **FAISS** — Local vector similarity search
- **Google Gemini API** — Embeddings + LLM (gemini-2.5-flash)
- **Streamlit** — Interactive web UI

## 🚀 How to Run
```bash
git clone https://github.com/Astitva028/rag-doc-qa
cd rag-doc-qa
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Add your Gemini API key to a `.env` file:

Then run:
```bash
streamlit run app.py
```

## 💡 How It Works
1. PDF is parsed and split into 1000-character overlapping chunks
2. Each chunk is embedded using Gemini `gemini-embedding-001`
3. Embeddings stored in a local FAISS index
4. User question is embedded and top-4 similar chunks are retrieved
5. Gemini `gemini-2.5-flash` generates a grounded answer with page citations

## 📸 Demo
Upload any PDF → ask questions → get answers with source page numbers.