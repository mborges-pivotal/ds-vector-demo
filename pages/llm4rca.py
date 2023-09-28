import streamlit as st

st.set_page_config(page_title="llm4rca", page_icon="📖")

st.sidebar.header("Large-Language models for automatic incident management")

st.header("LLM for RCA")
st.write(
"""
This is base on the blog [Large-language models for automatic cloud incident management](https://www.microsoft.com/en-us/research/blog/large-language-models-for-automatic-cloud-incident-management/) 
by Microsoft that proposes adapting large-language models for automated incident management. 
"""
)

st.markdown(
    '<img src="../app/static/llm2rca.jpeg" height="400" style="padding-bottom: 50% border: 5px solid orange">',
    unsafe_allow_html=True,
)
