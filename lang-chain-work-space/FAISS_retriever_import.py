from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Load persisted FAISS index
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Convert vector store into retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    return retriever
