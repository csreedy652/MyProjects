from dotenv import load_dotenv
import os
load_dotenv()
print(os.environ["OPENAI_API_KEY"])
# from openai import OpenAI
# client = OpenAI()

# response = client.responses.create(
#     model="gpt-4.1",
#     input="Write a one-sentence bedtime story about a unicorn."
# )

# print(response.output_text)
user_question=input("Enter your Question: ")
from langchain.prompts import PromptTemplate
text="""You are a Tollywood Fancy Chatbot. Always reply with one tollywood dialouge.
Below is user question: 
{question}
"""
prompt=PromptTemplate(
    input_variables=["question"],
    template=text
)
# from langchain_openai import ChatOpenAI
# llm=ChatOpenAI(model="gpt-4o")
from langchain_groq import ChatGroq
llm=ChatGroq(model="deepseek-r1-distill-llama-70b")
chain=prompt | llm
result=chain.invoke({"question":user_question})
print(result.content)