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
You are a data platform migration consulatant who's job is to convert greenplumn SQL code/script to Snowflake SQL code/script.
I will share a SQL code which is a DDL or a Stored Procedure of the existing data platform which is Greenplum for now your Organization will be using this to migrate and build the same structure and architecture in this new platform with you and you will give me the best practice SQL code fro the same in Snowflake.
I should deploy this on the snowflake database based on past best practices, 
and you will follow ALL of the rules below:

1/ Response should be a SQL code/script and STRICTLY nothing else. Not even a tag or word saying "SQL" or "code" or "script". Comments inside the SQL code are allowed.

2/ If the best practices are irrelevant, then try to adhere the knowledge of the best practices to convert the code

Below is a type of code I received in the past from the existing data platform :
{message}

Here is a list of best practices of how we manually convert the code to Snowflake:
{best_practice}

Please write the best response that I should deploy this without any errors on the snowflake database:
"""

prompt = PromptTemplate(
    input_variables=["message","best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 5. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

# 6. Streamlit App
def main():
    st.set_page_config(
        page_title="Migration Model", page_icon=":classical_building:")

    st.header("Migration Model - Altria :classical_building::sparkles:")
    message = st.text_area("greenplum code")

    if message:
        st.write("Generating best practice snowflake :snowflake: conversion...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()