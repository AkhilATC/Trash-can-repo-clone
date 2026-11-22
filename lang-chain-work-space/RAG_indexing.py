# INDEXING
"""
Indexing your documents, that is, preprocessing them in a way where your
application can easily find the most relevant ones for each question
steps:
    1. Extract the text from the document.
    2. Split the text into manageable chunks.
    3. Convert the text into numbers that computers can understand.
    4. Store these number representations of your text somewhere that makes it easy
    and fast to retrieve the relevant sections of your document to answer a given
    question.
"""
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# vector Db connections
from langchain_postgres.vectorstores import PGVector


def text_loader():
    loader = TextLoader("sample.txt")
    doc = loader.load()
    print(doc)


def pdf_loader():
    file_name = "rag_pdf_document_best_practices_guide_and_sample.pdf"
    loader = PyPDFLoader(file_name)
    doc = loader.load()
    # print(doc)
    return doc

def spliting_based_on_char(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splitted_docs = splitter.split_documents(docs)
    # print(splitted_docs)
    return splitted_docs

# spliting based on token
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=75
)

chunks = splitter.split_documents(docs)
"""

def create_embeddings(splitted_docs):
    """
    embed_documents()	Generate embeddings for document chunks
    embed_query()	Generate embedding for a user's search query
    :param splitted_docs:
    :return:
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectors = embeddings.embed_documents(
        [doc.page_content for doc in splitted_docs]
    )

    print("Total documents:", len(splitted_docs))
    print("Total embeddings:", len(vectors))
    print("Vector dimensions:", len(vectors[0]))

    return vectors,embeddings

def save_to_vector_db(vectors,embeddings_model):
    #pgvector = postgresql://postgres:<POSTGRES_PASSWORD>@localhost:5432/postgres
    connection = "postgresql://postgres:ChangeThis_Postgres123!@localhost:5432/postgres"
    pg_vectors = PGVector.from_documents(vectors,embeddings_model,connection=connection)
    return pg_vectors




if __name__ == "__main__":
    docs = pdf_loader()
    splited_docs = spliting_based_on_char(docs)
    vectors,embedding_model = create_embeddings(splited_docs)

    db = save_to_vector_db(vectors,embedding_model)