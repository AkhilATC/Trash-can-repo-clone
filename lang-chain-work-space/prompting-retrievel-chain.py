from langchain_core.prompts import ChatPromptTemplate
from llm_import import llm
from FAISS_retriever_import import get_retriever
from langchain_core.runnables import chain

retriever = get_retriever()
prompt = ChatPromptTemplate.from_template("""Answer the question based only on
    the following context:
   {context}
   Question: {question}
   """)


@chain
def execute_chain(q):
    # Retrieve relevant documents
    docs = retriever.invoke(q)

    # Extract text from retrieved documents
    context = "\n\n".join(
        doc.page_content for doc in docs
    )
    prompt_invoked = prompt.invoke({"context": context,"question": q})

    answer = llm.invoke(prompt_invoked)
    return answer

r = execute_chain.invoke("what are the RAG pipelines?")
print(r)