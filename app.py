import warnings

# Suppress all deprecation warnings globally
warnings.simplefilter("ignore", DeprecationWarning)

import streamlit as st
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
    
for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)

load_dotenv(override=True)

# 1. Load the knowledge base
loader = JSONLoader(file_path="knowledge.json", jq_schema='.[] | {topic, greenplum, snowflake}', text_content=False)
documents = loader.load()

print("Length of documents -", len(documents))
print("First document -", documents[0])

# 2. Vectorize the knowledge base
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 3. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    for idx, content in enumerate(page_contents_array, start=1):
        print(f"{idx:02d} - {content}")
    return page_contents_array

# 4. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")

template = """
You are a data platform migration consultant whose job is to convert Greenplum SQL code/script to Snowflake SQL code/script.
I will share a SQL code which is a DDL or a Stored Procedure of the existing data platform which is Greenplum for now your Organization will be using this to migrate and build the same structure and architecture in this new platform with you and you will give me the best practice SQL code for the same in Snowflake.
I should deploy this on the Snowflake database based on past best practices, 
and you will follow ALL of the rules below:

1/ Response should be a SQL code/script and STRICTLY nothing else. Not even a tag or word saying "SQL" or "code" or "script". Comments inside the SQL code are allowed.

2/ If the best practices are irrelevant, then try to adhere to the knowledge of the best practices to convert the code

Below is a type of code I received in the past from the existing data platform:
{message}

Here is a list of best practices of how we manually convert the code to Snowflake:
{best_practice}

Please write the best response that I should deploy this without any errors on the Snowflake database:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 5. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

def on_submit():
    st.session_state["show_loading"] = True
    st.session_state["show_result"] = False

# 6. Streamlit App
def main():
    st.set_page_config(
        page_title="Migration Model", page_icon=":classical_building:")

    st.header("Migration Model - Altria :classical_building::sparkles:")

    message = st.text_area("greenplum code", key="message")

    # Add empty space before buttons
    st.write("")
    
    # Add square buttons
    text1, text2, col1, col2, col3, col4, col5, col6 = st.columns([1, 0.7, 0.15, 0.15, 0.15, 0.15, 0.15, 0.4])
    with text1:
        st.write("")
    with text2:
        st.write("Example codes :point_right:")
    with col1:
        st.button("1", key="btn1")
    with col2:
        st.button("2", key="btn2")
    with col3:
        st.button("3", key="btn3")
    with col4:
        st.button("4", key="btn4")
    with col5:
        st.button("5", key="btn5")
    with col6:
        st.button("Enter", key="submit", type="primary", on_click=on_submit)

    # Add empty space after buttons
    st.write("")

    if message:
        on_submit()

    # Display result if available
    if "show_result" not in st.session_state or "show_loading" not in st.session_state:
        st.session_state["show_loading"] = False
        st.session_state["show_result"] = False

    if st.session_state["show_loading"]:
        st.write("Generating best practice snowflake :snowflake: conversion...")
        message = st.session_state["message"]
        if message:
            result = generate_response(message)
            st.session_state["result"] = result
            st.session_state["show_result"] = True
            st.session_state["show_loading"] = False

    if st.session_state["show_result"]:
        st.info(st.session_state["result"])

if __name__ == '__main__':
    main()