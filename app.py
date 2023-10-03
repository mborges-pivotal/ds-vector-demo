import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ts_vstore import load_timeseries, get_train_test_split, get_portion_to_embed, get_plot_data, plot_arrays, get_sliding_window, get_step
from agent_memory import get_answer, format_messages, clear_memory, load_memory
from rca_vstore import load_support_tickets, sim_search

# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
"""
See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df['dummy2'] = 0
/Users/marceloborges/datastax/customers/dish/ts-vs-ui/app.py:46: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead
"""
pd.options.mode.chained_assignment = None  # default='warn'

#Globals
cqlMode = 'astra_db'

"""MAIN Function
"""
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    import os
    load_dotenv(find_dotenv(), override=True)
    ASTRA_DB_KEYSPACE = os.environ["ASTRA_DB_KEYSPACE"]

    # Split the data into training and testing sets
    train_data, test_data, test_data_size, data_size = get_train_test_split()

    # Define x coordinate for the vertical line to show which part was used for embedding
    vertical_line_x = get_portion_to_embed()

    # Define colors for the arrays, first one in red and others in fading-out shades of blue
    colors = ['red', 'blue', 'dodgerblue', 'steelblue', 'skyblue', 'powderblue']
    labels = [
        'Query series',
        '#1 Closest neighbour',
        '#2 Closest neighbour',
        '#3 Closest neighbour',
        '#4 Closest neighbour',
        '#5 Closest neighbour',
    ]

    ########################
    # UI
    ########################
    st.title(':blue[Astra VectorDB] _Demonstration_')

    tab1, tab2, tab3 = st.tabs(["time series vectors", "Chat memory", "Support tickets"])

    st.sidebar.button('Load Time Series data', on_click=load_timeseries, args=[])
    st.sidebar.button('Load Support Ticket data', on_click=load_support_tickets, args=[])


    with tab1:
        st.subheader('Time Series Forecasting')
        col1, col2 = st.columns(2)
        with col1:
            st.write("dataset containing ", data_size, " entries.", "Vectorized ", len(train_data), " and using ", len(test_data), " to test.")
            st.write("")
            point = st.slider("Test data starting index", 0, test_data_size)
            # plot the array
            # TODO: get point from session state
            arrays = get_plot_data(point)
            fig = plot_arrays(arrays, vertical_line_x, colors, labels)
            st.dataframe(test_data)
        with col2:  
            st.write("Working with a ", get_sliding_window(), " hours sliding windows capturing data every ", get_step(), " hour.")      
            st.pyplot(fig)
            st.write("Vectorized the first ", get_portion_to_embed(), " hours of the sliding window and using the remainder to check the timeseries trend.")

    with tab2:
        # MMB: on_change should remove the llm answer, summary and memory text areas
        conversation_id = st.text_input(
            'Conversation ID', 'my-conv-id-01')

        col1, col2 = st.columns(2)
        with col1:
            clear_data = st.button(
                'Clear History', on_click=clear_memory, args=[conversation_id])
        with col2:
            load_data = st.button(
                'Load Conversation Memory', on_click=load_memory, args=[conversation_id])

        q = st.text_input("Message")
        if q:
            answer = get_answer(conversation_id, q)
            st.text_area('LLM Answer: ', value=answer)

        if 'summary' in st.session_state:
            st.divider()
            st.text_area(label=f"Summary for conversation id: {st.session_state.conversation_id}", value=st.session_state.summary, height=200, key='summary')
        
        if 'messages' in st.session_state:
            st.divider()
            st.text_area(label="Memory", value=format_messages(
                st.session_state.messages), height=400, key='memory')

    with tab3:
        st.write("Search support tickets for past RCA and use in a fine tuned model for propose resolution and mitigation plan")
        q = st.text_input('Message', key='q')
        if q:
            docs = sim_search(q)
            for i, doc in enumerate(docs):
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.text_input('Type', doc.metadata['Ticket Type'], key="type_%d"%i, disabled=True)
                with col2:
                    st.text_input('Rating', doc.metadata['Customer Satisfaction Rating'], key="rating_)%d"%i, disabled=True)
                st.text_input('Summary:', doc.metadata['Ticket Subject'], key="summary_%d"%i, disabled=True)
                st.text_area('Description', doc.metadata['Ticket Description'], key="description_%d"%i)
                st.text_area('Resolution', doc.metadata['Resolution'], key="resolution_%d"%i)
