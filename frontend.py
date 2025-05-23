import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
load_dotenv()
st.set_page_config(page_title="ChatBot", layout="centered")
st.title("My First ChatBot")
st.markdown("Ask Any Questions and Get Instant reply")
user_question=st.text_input("Ask your Question")
if st.button("Get Answer") and user_question.strip():
    with st.spinner("Fetching the most updated answer..."):
        text="""You are a Tollywood Fancy Chatbot. Always reply with one tollywood dialouge.
Below is user question: 
{question}
"""
        prompt=PromptTemplate(
    input_variables=["question"],
    template=text
)
        from langchain_groq import ChatGroq
        llm=ChatGroq(model="deepseek-r1-distill-llama-70b")
        chain=prompt | llm
        try:
            result=chain.invoke({"question":user_question})
            st.success("Here is your Answer")
            st.write(result.content)
        except Exception as e:
            st.error(f"something went wrong: {str(e)} ")

else:
    st.caption("Powered by Mac")  