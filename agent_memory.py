import streamlit as st

from langchain.memory import CassandraChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

# Globals
table_name='astra_agent_memory'
llm = OpenAI()

""" clear_memory
"""
def clear_memory(session, keyspace, conversation_id):
    message_history = CassandraChatMessageHistory(
        session_id=conversation_id,
        session=session,
        keyspace=keyspace,
        ttl_seconds=3600,
        table_name=table_name
    )

    message_history.clear()

    if "conversation_id" in st.session_state:
        del st.session_state['conversation_id']
    if "messages" in st.session_state:
        del st.session_state['messages']
    if "summary" in st.session_state:
        del st.session_state['summary']
    return True

""" start_memory
"""
def start_memory():
    load_memory(st.session_state["conv_id_input"])
    return True

""" get_answer
"""
def get_answer(session, keyspace, conversation_id, q):
    st.session_state.conversation_id = conversation_id

    message_history = CassandraChatMessageHistory(
        session_id=conversation_id,
        session=session,
        keyspace=keyspace,
        ttl_seconds=3600,
        table_name=table_name

    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=message_history,
        max_token_limit=180,
        buffer=""
    )

    summaryConversation = ConversationChain(
        llm=llm,
        memory=memory

    )

    answer = summaryConversation.predict(input=q)
    print("Full answer")
    print(answer)

    new_summary = memory.predict_new_summary(
        memory.chat_memory.messages,
        memory.moving_summary_buffer,
    )

    st.session_state.messages = memory.chat_memory.messages
    st.session_state.summary = new_summary

    return answer

""" load_memory
"""
def load_memory(session, keyspace, conversation_id):
    st.session_state.conversation_id = conversation_id

    message_history = CassandraChatMessageHistory(
        session_id=conversation_id,
        session=session,
        keyspace=keyspace,
        ttl_seconds=3600,
        table_name=table_name
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=message_history,
        max_token_limit=180,
        buffer=""
    )

    new_summary = memory.predict_new_summary(
        memory.chat_memory.messages,
        memory.moving_summary_buffer,
    )

    st.session_state.messages = memory.chat_memory.messages
    st.session_state.summary = new_summary

    return memory.chat_memory.messages, new_summary

""" format_messages
"""
def format_messages(messages):
    res = ""
    for m in reversed(messages):
        res += f'{type(m).__name__}: {m.content}\n\n'
    return res
