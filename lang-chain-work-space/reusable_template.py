from llm_import import llm
from langchain_core.prompts import PromptTemplate


template = PromptTemplate.from_template("""Answer the question based on the
 context below. If the question cannot be answered using the information
 provided, answer with "I don't know".
Context: {context}
Question: {question}
Answer: """)

prompt = template.invoke({
 "context": """The most recent advancements in NLP are being driven by Large
 Language Models (LLMs). These models outperform their smaller
 counterparts and have become invaluable for developers who are creating
 applications with NLP capabilities. Developers can tap into these
 models through Hugging Face's `transformers` library, or by utilizing
 OpenAI and Cohere's offerings through the `openai` and `cohere`
 libraries, respectively.""",
 "question": "Which model providers offer LLMs?"
})
response = llm.invoke(prompt)
print("RESPONSE: >",response.content)

## LCEL MODEL
# chain = template | llm
#
# response = chain.invoke({
#     "context": "LLMs are offered by Hugging Face, OpenAI and Cohere.",
#     "question": "Which model providers offer LLMs?"
# })
# print(response.content)