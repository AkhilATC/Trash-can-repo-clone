
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llm_import import llm
from langchain_core.prompts import ChatPromptTemplate



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

if __name__ == "__main__":
    retriever = get_retriever()
    prompt = ChatPromptTemplate.from_template("""Answer the question based only on
     the following context:
    {context}
    Question: {question}
    """)
    q = "what are the RAG pipelines?"
    chain = prompt | llm
    # Retrieve relevant documents
    docs = retriever.invoke(q)

    # Extract text from retrieved documents
    context = "\n\n".join(
        doc.page_content for doc in docs
    )


    r = chain.invoke({"context":context, "question":q})
    print(r.content)