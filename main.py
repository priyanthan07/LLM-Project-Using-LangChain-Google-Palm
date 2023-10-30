import streamlit as st 
from Langchain import get_qa_chain, create_vector_DB

st.title(" 🍁    FAQ Assistant      🍁")
button = st.button("Create base")      # this should be available for developers only

if button:
    pass

question = st.text_input("Question: ")

submit = st.button(" Submit ")


if question:
    chain = get_qa_chain()
    response = chain(question)
    if submit:
        st.header("Answer: ")
        st.write(response["result"])
