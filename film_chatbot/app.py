import streamlit as st
import os
import time
from rag import initialize_rag, build_vectorstore, load_documents, VECTORSTORE_PATH

st.set_page_config(
    page_title="🎬 Film Expert Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&display=swap');

    .stApp { background-color: #0d0d0d; color: #f0f0f0; }

    .hero {
        background: linear-gradient(135deg, #1a0a00 0%, #0d0d0d 60%, #000d1a 100%);
        border: 1px solid #ff6600;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3rem;
        color: #ff6600;
        letter-spacing: 3px;
        margin: 0 0 0.25rem 0;
    }
    .hero p { color: #aaa; font-family: 'Inter', sans-serif; margin: 0; }

    .chat-user {
        background: #1a1a2e;
        border-left: 3px solid #ff6600;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-family: 'Inter', sans-serif;
    }
    .chat-bot {
        background: #111;
        border-left: 3px solid #00b4d8;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-family: 'Inter', sans-serif;
    }
    .source-tag {
        display: inline-block;
        background: #222;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #888;
        margin: 2px;
    }
    .stat-box {
        background: #111;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stat-num { font-size: 1.8rem; color: #ff6600; font-weight: 700; }
    .stat-label { font-size: 0.8rem; color: #888; }

    .lang-badge {
        display: inline-block;
        background: #1a1a2e;
        border: 1px solid #ff6600;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #ff6600;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stChatInput"] textarea {
        background: #1a1a1a !important;
        color: #f0f0f0 !important;
        border: 1px solid #ff6600 !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: #ff6600;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover { background: #e55a00; }
</style>
""", unsafe_allow_html=True)

# ── Language config ───────────────────────────────────────────
LANGUAGES = {
    "🇬🇧 English": {
        "code": "en",
        "prompt_instruction": "You must respond in English.",
        "sample_qs": [
            "Who directed Inception?",
            "What is the plot of Interstellar?",
            "Tell me about The Godfather",
            "What movies did Tarantino direct?",
            "What happened in Back to the Future?",
        ],
        "chat_placeholder": "Ask about any film, director, or cinema topic...",
        "thinking": "🎬 Thinking...",
        "loading": "🎬 Loading Film Expert AI...",
        "subtitle": "AI-powered chatbot with RAG architecture — ask anything about films, directors & cinema history",
        "you": "You",
        "expert": "Film Expert",
        "docs_label": "Documents Loaded",
        "questions_label": "Questions Asked",
        "clear": "🗑️ Clear conversation",
        "rebuild": "🔄 Rebuild Vectorstore",
    },
    "🇭🇷 Hrvatski": {
        "code": "hr",
        "prompt_instruction": "Moraš odgovoriti na hrvatskom jeziku. Cijeli odgovor mora biti na hrvatskom.",
        "sample_qs": [
            "Tko je režirao Inception?",
            "O čemu govori film Interstellar?",
            "Reci mi nešto o Kumu (The Godfather)",
            "Koje filmove je snimio Tarantino?",
            "Što se događa u Povratak u budućnost?",
        ],
        "chat_placeholder": "Pitaj o filmovima, redateljima ili povijesti kina...",
        "thinking": "🎬 Razmišljam...",
        "loading": "🎬 Učitavam Film Expert AI...",
        "subtitle": "AI chatbot s RAG arhitekturom — pitaj bilo što o filmovima, redateljima i povijesti kina",
        "you": "Ti",
        "expert": "Film Ekspert",
        "docs_label": "Učitani dokumenti",
        "questions_label": "Postavljenih pitanja",
        "clear": "🗑️ Očisti razgovor",
        "rebuild": "🔄 Ponovno izgraditi Vectorstore",
    },
    "🇩🇪 Deutsch": {
        "code": "de",
        "prompt_instruction": "Du musst auf Deutsch antworten. Die gesamte Antwort muss auf Deutsch sein.",
        "sample_qs": [
            "Wer hat Inception gedreht?",
            "Worum geht es in Interstellar?",
            "Erzähl mir über Der Pate",
            "Welche Filme hat Tarantino gedreht?",
            "Was passiert in Zurück in die Zukunft?",
        ],
        "chat_placeholder": "Frag nach Filmen, Regisseuren oder Kinogeschichte...",
        "thinking": "🎬 Denke nach...",
        "loading": "🎬 Film Expert AI wird geladen...",
        "subtitle": "KI-Chatbot mit RAG-Architektur — frag alles über Filme, Regisseure und Kinogeschichte",
        "you": "Du",
        "expert": "Film Experte",
        "docs_label": "Geladene Dokumente",
        "questions_label": "Gestellte Fragen",
        "clear": "🗑️ Gespräch löschen",
        "rebuild": "🔄 Vectorstore neu erstellen",
    },
}

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 Language / Jezik / Sprache")
    selected_lang = st.selectbox(
        "Choose language:",
        list(LANGUAGES.keys()),
        index=0,
        label_visibility="collapsed"
    )
    lang = LANGUAGES[selected_lang]

    if "current_lang" not in st.session_state:
        st.session_state.current_lang = selected_lang
    elif st.session_state.current_lang != selected_lang:
        st.session_state.current_lang = selected_lang
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(f"### ⚙️ Controls")
    if st.button(lang["rebuild"], use_container_width=True):
        if os.path.exists(VECTORSTORE_PATH):
            import shutil
            shutil.rmtree(VECTORSTORE_PATH)
        with st.spinner("Building vectorstore..."):
            docs = load_documents("documents")
            build_vectorstore(docs)
        st.session_state.rag_chain = None
        st.success("Done!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 📂 Documents")
    doc_path = "documents"
    if os.path.exists(doc_path):
        for f in sorted(os.listdir(doc_path)):
            ext = f.split(".")[-1].upper()
            color = "#ff6600" if ext == "PDF" else "#00b4d8"
            st.markdown(f'<span style="color:{color}">[{ext}]</span> {f}', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    for q in lang["sample_qs"]:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.prefill = q

# ── Init RAG ─────────────────────────────────────────────────
if "rag_chain" not in st.session_state or st.session_state.rag_chain is None:
    with st.spinner(lang["loading"]):
        st.session_state.rag_chain = initialize_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Hero ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <h1>🎬 FILM EXPERT</h1>
    <p>{lang["subtitle"]}</p>
</div>
""", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
doc_count = len(os.listdir("documents")) if os.path.exists("documents") else 0
msg_count = len(st.session_state.messages)
with col1:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{doc_count}</div><div class="stat-label">{lang["docs_label"]}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{msg_count // 2}</div><div class="stat-label">{lang["questions_label"]}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-box"><div class="stat-num">RAG</div><div class="stat-label">Architecture</div></div>', unsafe_allow_html=True)

st.markdown(f'<div class="lang-badge">🌍 {selected_lang}</div>', unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">🧑 <strong>{lang["you"]}:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bot">🤖 <strong>{lang["expert"]}:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        if "sources" in msg and msg["sources"]:
            src_html = "".join([f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"]])
            st.markdown(f"<div style='margin-top:4px'>{src_html}</div>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input(lang["chat_placeholder"])

if prefill:
    user_input = prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner(lang["thinking"]):
        start = time.time()
        # Inject language instruction into query
        query_with_lang = f"{lang['prompt_instruction']}\n\nQuestion: {user_input}"
        result = st.session_state.rag_chain.invoke({"query": query_with_lang})
        elapsed = time.time() - start

    answer = result["result"]
    sources = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            src = doc.metadata.get("source", "")
            name = os.path.basename(src) if src else "unknown"
            if name not in sources:
                sources.append(name)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "time": round(elapsed, 1)
    })
    st.rerun()

# ── Clear ─────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button(lang["clear"]):
        st.session_state.messages = []
        st.rerun()