import streamlit as st

st.set_page_config(page_title="logevent2vec", page_icon="📖")

st.sidebar.header("LogEvent-to-Vector Based Anomaly Detection")

st.header("LogEvent2vec: LogEvent-to-Vector Based Anomaly Detection for Large-Scale Logs in Internet of Things")
st.write(
"""
The [LogEvent2vec: LogEvent-to-Vector Based Anomaly Detection for Large-Scale Logs in Internet of Things](https://www.mdpi.com/1424-8220/20/9/2451) research paper 
proposes log specific variation of word2vec reduces computational time and improve accuracy. We found an [implementation](https://github.com/NetManAIOps/Log2Vec) of the model. 

This other blog from SienceLogic, [Using GPT-3 for plain language incident root cause from logs](https://sciencelogic.com/blog/using-gpt-3-for-plain-language-incident-root-cause-from-logs)
uses logs for deriving RCA from logs
"""
)

st.markdown(
    '<img src="../app/static/logevent2vec.png" width="850 height=400" style="padding-bottom: 50%">',
    unsafe_allow_html=True,
)
