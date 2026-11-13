from llm_import import llm
from pydantic import BaseModel,Field


class AnswerWithJustification(BaseModel):
    answer: str = Field(description="Answer to the question")
    justification: str = Field(description="Reason for the answer")

structured_llm = llm.with_structured_output(AnswerWithJustification)
response = structured_llm.invoke("""What weighs more, a pound of bricks or a pound
 of feathers""")
print(response.content)
