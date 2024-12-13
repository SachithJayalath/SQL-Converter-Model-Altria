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

def return_example(idx):
    examples = {
        1: "CREATE TABLE subscribers (subscriber_id SERIAL PRIMARY KEY, subscriber_name TEXT, subscription_plan TEXT CHECK (subscription_plan IN ('basic', 'premium', 'enterprise')), subscription_start DATE DEFAULT CURRENT_DATE, subscription_end DATE) DISTRIBUTED BY (subscriber_id);",
        2: "CREATE TABLE billing (bill_id SERIAL PRIMARY KEY, subscriber_id INT, billing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, total_amount NUMERIC CHECK (total_amount >= 0)) PARTITION BY RANGE (billing_date) (START ('2023-01-01') END ('2024-01-01') EVERY ('1 MONTH')) DISTRIBUTED BY (bill_id);",
        3: "CREATE EXTERNAL TABLE sms_logs (message_id INT, sender_number TEXT, receiver_number TEXT, timestamp TIMESTAMP, message_status TEXT CHECK (message_status IN ('sent', 'failed'))) LOCATION ('s3://telecom-dw/sms_logs/') FORMAT 'CSV' LOG ERRORS;",
        4: "CREATE TABLE network_events (event_id SERIAL PRIMARY KEY, event_type TEXT CHECK (event_type IN ('outage', 'maintenance', 'upgrade')), region TEXT, event_start TIMESTAMP NOT NULL, event_end TIMESTAMP, duration INTERVAL GENERATED ALWAYS AS (event_end - event_start)) DISTRIBUTED BY (region);",
        5: "CREATE OR REPLACE FUNCTION add_transaction(p_customer_id INT, p_transaction_amount NUMERIC, p_payment_method TEXT, p_transaction_status TEXT, p_discount_applied BOOLEAN DEFAULT FALSE) RETURNS VOID AS $$BEGIN INSERT INTO customer_transactions(customer_id, transaction_date, transaction_amount, payment_method, transaction_status, discount_applied) VALUES(p_customer_id, CURRENT_TIMESTAMP, p_transaction_amount, p_payment_method, p_transaction_status, p_discount_applied);END;$$ LANGUAGE plpgsql;"
    }
    st.session_state["message"] = examples.get(idx, "")
    st.session_state["show_result"] = False

# 6. Streamlit App
def main():
    st.set_page_config(
        page_title="Migration Model", page_icon=":classical_building:")

    st.header("Migration Model - Altria :classical_building::sparkles:")

    if "message" not in st.session_state:
        st.session_state["message"] = ""

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
        st.button("1", key="btn1", on_click=lambda: return_example(1))
    with col2:
        st.button("2", key="btn2", on_click=lambda: return_example(2))
    with col3:
        st.button("3", key="btn3", on_click=lambda: return_example(3))
    with col4:
        st.button("4", key="btn4", on_click=lambda: return_example(4))
    with col5:
        st.button("5", key="btn5", on_click=lambda: return_example(5))
    with col6:
        st.button("Enter", key="submit", type="primary", on_click=on_submit)

    # Add empty space after buttons
    st.write("")

    if st.session_state["message"] and st.session_state["show_loading"]:
        st.write("Generating best practice snowflake :snowflake: conversion...")
        message = st.session_state["message"]
        if message:
            result = generate_response(message)
            st.session_state["result"] = result
            st.session_state["show_result"] = True
            st.session_state["show_loading"] = False

    # Display result if available
    if "show_result" not in st.session_state or "show_loading" not in st.session_state:
        st.session_state["show_loading"] = False
        st.session_state["show_result"] = False

    if st.session_state["show_result"]:
        st.info(st.session_state["result"])

if __name__ == '__main__':
    main()