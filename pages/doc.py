import streamlit as st
from PIL import Image


st.set_page_config(page_title="Documentation", page_icon="📖")

st.markdown("# Documentation")
st.sidebar.header("Documentation")

st.subheader("Time Series Forecasting - Time2Vec")
st.write(
"""
Extracting value from the time series gets complicated really quickly. Yes, one can plug them into a monitoring
stack and get some visualisations. Going further, into analytics, ML and forecasting, things become far less 
easy. We suddenly need data scientist, ML engineers and a whole lot of computational resources.

The [Time2Vec: Learning a Vector Representation of Time](https://arxiv.org/pdf/1907.05321.pdf) research paper 
proposes a time series vector embedding model with many applications. This demo is based on [this](https://github.com/ojus1/Date2Vec) implementation
of it. 
"""
)

st.markdown(
    '<img src="../app/static/workflow.png" height="400" style="padding-bottom: 50% border: 5px solid orange">',
    unsafe_allow_html=True,
)

# st.markdown("![workflow](../app/static/workflow.png)")

st.write(
"""
##### Electricity Sample Data
Below is the AstraDB table we used. Notice that sliding windows and vector size are based on the granularity
of your embeddings
"""
)


st.code('''
CREATE TABLE IF NOT EXISTS {ASTRA_KEYSPACE_NAME}.electricity (
  id text PRIMARY KEY,
  orig_timestamps vector<float, {SLIDING_WINDOW_SIZE}>,
  orig_demand VECTOR<float, {SLIDING_WINDOW_SIZE}>,
  orig_temperature VECTOR<float, {SLIDING_WINDOW_SIZE}>,
  embedding VECTOR<float, {VECTOR_SIZE}>,  
);
        ''')