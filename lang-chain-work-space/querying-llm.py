
# INDEXING
"""
Indexing steps:
1. Extract text from documents.
2. Split text into manageable chunks.
3. Generate embeddings.
4. Store embeddings in FAISS for similarity search.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def pdf_loader():
    file_name = "rag_pdf_document_best_practices_guide_and_sample.pdf"
    loader = PyPDFLoader(file_name)
    docs = loader.load()
    return docs


def splitting_based_on_char(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    splitted_docs = splitter.split_documents(docs)
    return splitted_docs


def create_embeddings():
    """
    embed_documents() -> Embeddings for document chunks
    embed_query()     -> Embedding for user queries
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


def save_to_vector_db(splitted_docs, embeddings_model):

    # Create FAISS vector store
    faiss_db = FAISS.from_documents(
        documents=splitted_docs,
        embedding=embeddings_model
    )

    # Persist FAISS index locally
    faiss_db.save_local("faiss_index")

    print("Documents indexed:", len(splitted_docs))
    print("Vector dimensions:", len(
        embeddings_model.embed_query("Sample query")
    ))

    return faiss_db

def get_retriever(embeddings):



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

if __name__ == "__main__":

    docs = pdf_loader()

    splitted_docs = splitting_based_on_char(docs)

    embedding_model = create_embeddings()

    db = save_to_vector_db(
        splitted_docs,
        embedding_model
    )

    print("FAISS indexing completed!")

    # retriever definition
    # retriever = db.as_retriever()
    # relevent_docs = retriever.invoke("""Understand the RAG document pipeline""")

    # get retriever = get_retriever() from a defined function
    retriever = get_retriever(embedding_model)
    retrieved_docs = retriever.invoke("""Understand the RAG document pipeline""")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nDocument {i + 1}")
        print(doc.page_content)
        print("Metadata:", doc.metadata)