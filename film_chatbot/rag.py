import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"

def load_documents(path: str):
    docs = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
                docs.extend(loader.load())
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    print(f"Loaded {len(docs)} document pages from {path}")
    return docs

def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vectorstore saved to {VECTORSTORE_PATH}")
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def get_all_chunks():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vs = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    index = vs.index
    docstore = vs.docstore
    id_map = vs.index_to_docstore_id

    texts = []
    vectors = []
    import numpy as np
    n = index.ntotal
    raw_vectors = index.reconstruct_n(0, n)
    for i in range(n):
        doc_id = id_map.get(i)
        if doc_id:
            doc = docstore.search(doc_id)
            texts.append(doc.page_content[:80])
            vectors.append(raw_vectors[i])
    return texts, np.array(vectors)

def build_rag_chain(vectorstore):
    llm = OllamaLLM(model="llama3.2", temperature=0.1)

    prompt = PromptTemplate(
        template="""You are a knowledgeable film expert assistant. Use the following context from film documents to answer the question accurately and helpfully.

Context:
{context}

Question: {question}

Answer based on the context provided. If the context doesn't contain enough information, say so honestly but still try to help with your general knowledge.""",
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    class RAGChain:
        def __init__(self):
            self.last_source_documents = []

        def invoke(self, inputs):
            query = inputs["query"]
            docs = retriever.invoke(query)
            self.last_source_documents = docs
            context = format_docs(docs)
            formatted_prompt = prompt.format(context=context, question=query)
            answer = llm.invoke(formatted_prompt)
            return {
                "result": answer,
                "source_documents": docs
            }

    return RAGChain()

def initialize_rag():
    if os.path.exists(VECTORSTORE_PATH):
        print("Loading existing vectorstore...")
        vectorstore = load_vectorstore()
    else:
        print("Building new vectorstore...")
        documents = load_documents(DOCUMENTS_PATH)
        vectorstore = build_vectorstore(documents)

    chain = build_rag_chain(vectorstore)
    return chain