# 🎬 Film Expert Chatbot

A local, specialized AI chatbot for film knowledge using RAG (Retrieval-Augmented Generation) architecture. Built as a university project for the course *Prikaz znanja i rezoniranje o znanju* (2025/26).

## Features

- 🤖 **Local LLM** via Ollama (llama3.2) — no internet or API required
- 📚 **RAG Architecture** — answers grounded in a curated film document collection
- 🗃️ **FAISS Vector Store** — fast semantic search over document chunks
- 🌍 **Multilingual** — English, Croatian, German
- 🖥️ **Streamlit GUI** — clean dark-themed web interface
- 📊 **Embedding Visualization** — PCA and t-SNE plots of document embeddings
- 📈 **Evaluation** — ROUGE and BLEU metrics

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed with `llama3.2` model

## Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd film_chatbot

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the LLM model
ollama pull llama3.2
```

## Usage

### Run the chatbot
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### Run evaluation
```bash
python evaluation.py
```
Outputs ROUGE and BLEU scores, saves results to `evaluation_results.json`

### Run embedding visualization
```bash
python visualize_embeddings.py
```
Saves `embedding_visualization.png` with PCA and t-SNE plots

## Project Structure

```
film_chatbot/
├── documents/          # Knowledge base (5x TXT + 5x PDF)
├── vectorstore/        # FAISS index (auto-generated)
├── app.py              # Streamlit web app
├── rag.py              # RAG pipeline (loading, chunking, embeddings, retrieval)
├── evaluation.py       # ROUGE + BLEU evaluation
├── visualize_embeddings.py  # PCA + t-SNE visualization
└── requirements.txt
```

## Documents / Knowledge Base

| File | Type | Topic |
|------|------|-------|
| 01_christopher_nolan.txt | TXT | Christopher Nolan |
| 02_quentin_tarantino.txt | TXT | Quentin Tarantino |
| 03_the_godfather.txt | TXT | The Godfather |
| 04_inception.txt | TXT | Inception (2010) |
| 05_interstellar.txt | TXT | Interstellar (2014) |
| Back_to_the_Future.pdf | PDF | Back to the Future |
| Demolition_Man_(film).pdf | PDF | Demolition Man |
| Pirates_of_the_Caribbean.pdf | PDF | Pirates of the Caribbean |
| Ready_Player_One_(film).pdf | PDF | Ready Player One |
| The_Naked_Gun.pdf | PDF | The Naked Gun |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama + llama3.2 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| UI | Streamlit |
| Evaluation | rouge-score, nltk (BLEU) |
| Visualization | matplotlib, scikit-learn |
