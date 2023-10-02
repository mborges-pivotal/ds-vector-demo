import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Cassandra

from langchain.llms import OpenAI

# Globals
table_name = 'vs_rca_openai'
llm = OpenAI()
embedding = OpenAIEmbeddings()

"""sim_search

Returns a list of k documents where k=3
"""
def sim_search(session, keyspace, text_embedding):

    cassandra_vstore = Cassandra(
        embedding=embedding,
        session=session,
        keyspace=keyspace,
        table_name=table_name,
    )

    return cassandra_vstore.search(text_embedding, search_type='similarity', k=3)