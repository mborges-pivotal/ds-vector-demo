import streamlit as st

st.set_page_config(page_title="time2vec", page_icon="📖")

st.sidebar.header("Time Series Forecasting")

st.header("Time Series Forecasting - Time2Vec")
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

# st.markdown("![workflow](../app/static/workflow.png)")
st.markdown(
    '<img src="../app/static/workflow.png" height="400" style="padding-bottom: 50% border: 5px solid orange">',
    unsafe_allow_html=True,
)

st.write("""
Let me walk you through the boxes and arrows:

We start with a Source of events.

* The events goes into Astra Streaming. At the very least, we sink it for archiving purposes, but we also send 
  it to a stream processing engine, such as Kaskada.
* The stream processor constructs time windows for us. For example, we can have 24 hourly readings per a 
  window covering one day.
* We can then pull this time window (or a part of it) through the embedding model and store the resulting 
  vector together with the original window.
* Later, we do a forecast by taking a not-yet complete window, get its embedding and doing a vector search 
  for the closes neighbours.
* We receive the windows that were the most similar to the one we’re querying with. It’s likely that the 
  current one will develop in a similar way.
""")

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