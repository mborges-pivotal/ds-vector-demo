import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Model import Date2VecConvert

from cqlsession import getCQLKeyspace, getCQLSession
from agent_memory import get_answer, format_messages, clear_memory, load_memory

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
session = getCQLSession(mode=cqlMode)
keyspace = getCQLKeyspace(mode=cqlMode)
table_name = 'electricity'

# For the real case, we take a 24h window
# Take the first 16h as the portion to embedd
# We do a step of every hour to get as many samples as possible
SLIDING_WINDOW_SIZE = 24
PORTION_TO_EMBED = 16
STEP = 1

# Load the Time2Vec model
d2v = Date2VecConvert(model_path="./d2v_model/d2v_98291_17.169918439404636.pth")

"""get_embedding_for_window
Using the model to create the embeddings for the time window
"""
def get_embedding_for_window(df):
    date_scaler = MinMaxScaler()
    demand_scaler = MinMaxScaler()
    temperature_scaler = MinMaxScaler()
    
    df[['date_n']] = date_scaler.fit_transform(df[['Date']].values.astype(np.int64))
    df[['Date']] = df[['date_n']]
    df[['demand_n']] = demand_scaler.fit_transform(df[['Demand']])
    df[['temperature_n']] = temperature_scaler.fit_transform(df[['Temperature']])

    # the Time2Vec model needs exactly 6-value long array as input
    # so we pad with 0s
    df['dummy1'] = 0
    df['dummy2'] = 0
    df['dummy3'] = 0    
    rows = df[['date_n', 'demand_n', 'temperature_n', 'dummy1', 'dummy2', 'dummy3']].values
    embedding = d2v(rows)

    flat_embeddings = [item.item() for sublist in embedding for item in sublist]

    return flat_embeddings

"""load_sample_data
Load the sample data used to train vectorize and test the model
"""
def load_sample_data():
    df = pd.read_parquet('./data/electricity/train-00000-of-00001.parquet', engine='pyarrow')

    data = df.sort_index(axis=0, ascending=True)

    data["Date"] = pd.to_datetime(data["__index_level_0__"])
    data.set_index('Date', inplace=True)
    data = data.reset_index().rename(columns={'index': 'Date'})
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.drop(columns=['__index_level_0__'], axis=1)

    return data

"""plot_array
helper function to plot the lines
"""
def plot_arrays(arrays, vertical_line_x, colors, labels):
    fig, ax = plt.subplots()
    for i, array in enumerate(arrays):
        ax.plot(array, color=colors[i], label=labels[i])
    ax.axvline(x=vertical_line_x, color='black', linestyle='--')  # Add vertical line
    ax.set_title(f'Forecasted Energy Consumption')
    ax.set_xlabel('Hours')
    ax.set_ylabel('Consumption')
    ax.legend()
    # plt.show()
    return plt

"""get_plot_data
Doing the actual vector search 
"""
def get_plot_data(test_data, point):
    
    # get the query time window
    q_df = test_data.iloc[point:point+SLIDING_WINDOW_SIZE, :]
    # take the first part of it to embedd
    q_wdf = q_df.head(PORTION_TO_EMBED)

    embedding = get_embedding_for_window(q_wdf)

    # do a nearest-neighbours query on the embedding
    q = f"""
    SELECT * FROM {ASTRA_DB_KEYSPACE}.electricity
    ORDER BY embedding ANN OF {embedding} LIMIT 5
    """
    rows = session.execute(q)

    # make simple lists of
    # original demands we got from the neighbours
    orig_demands = [row.orig_demand for row in rows]

    # the input demand we queried with
    query_demand = q_df[['Demand']].values.tolist()

    # concatenate the arrays for plotting purposes
    arrays = [query_demand] + orig_demands
    return arrays


"""MAIN Function
"""
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    import os
    load_dotenv(find_dotenv(), override=True)
    ASTRA_DB_KEYSPACE = os.environ["ASTRA_DB_KEYSPACE"]

    data = load_sample_data()

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    test_data_size = len(test_data) - SLIDING_WINDOW_SIZE

    # Define x coordinate for the vertical line to show which part was used for embedding
    vertical_line_x = PORTION_TO_EMBED

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
    st.header('Astra VectorDB Demonstration')

    # with st.sidebar:
    #     point = st.slider("Test data starting index", 0, test_data_size)
 
    tab1, tab2, tab3 = st.tabs(["time series vectors", "Chat memory", "Support tickets"])

    with tab1:
        st.subheader('Time Series Forecasting')
        col1, col2 = st.columns(2)
        with col1:
            st.write("dataset containing ", len(data), " entries.", "Vectorized ", len(train_data), " and using ", len(test_data), " to test.")
            st.write("")
            point = st.slider("Test data starting index", 0, test_data_size)
            # plot the array
            # TODO: get point from session state
            arrays = get_plot_data(test_data, point)
            fig = plot_arrays(arrays, vertical_line_x, colors, labels)
            st.dataframe(test_data)
        with col2:  
            st.write("Working with a ", SLIDING_WINDOW_SIZE, " hours sliding windows capturing data every ", STEP, " hour.")      
            st.pyplot(fig)
            st.write("Vectorized the first ", PORTION_TO_EMBED, " hours of the sliding window and using the remainder to check the timeseries trend.")

    with tab2:
        # MMB: on_change should remove the llm answer, summary and memory text areas
        conversation_id = st.text_input(
            'Conversation ID', 'my-conv-id-01')

        col1, col2 = st.columns(2)
        with col1:
            clear_data = st.button(
                'Clear History', on_click=clear_memory, args=[session, keyspace, conversation_id])
        with col2:
            load_data = st.button(
                'Load Conversation Memory', on_click=load_memory, args=[session, keyspace, conversation_id])

        q = st.text_input("Message")
        if q:
            answer = get_answer(session, keyspace ,conversation_id, q)
            st.text_area('LLM Answer: ', value=answer)

        if 'summary' in st.session_state:
            st.divider()
            st.text_area(label=f"Summary for conversation id: {st.session_state.conversation_id}", value=st.session_state.summary, height=200)
        
        if 'messages' in st.session_state:
            st.divider()
            st.text_area(label="Memory", value=format_messages(
                st.session_state.messages), height=400)

    with tab3:
        st.write("## Search support tickets for past RCA and use in a fine tuned model for propose resolution and mitigation plan")

    # with st.sidebar:
    #     conversation_id = st.text_input(
    #         'Conversation ID', 'my-conv-id-01')
    #     clear_data = st.button(
    #         'Clear History', on_click=clear_memory, args=[conversation_id])
    #     load_data = st.button(
    #         'Load Conversation Memory', on_click=load_memory, args=[conversation_id])