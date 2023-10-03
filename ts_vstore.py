import streamlit as st

from cqlsession import getCQLSession

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Model import Date2VecConvert


from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv(), override=True)
ASTRA_DB_KEYSPACE = os.environ["ASTRA_DB_KEYSPACE"]

#Globals
cqlMode = 'astra_db'
table_name = 'vs_rca_openai'

session = getCQLSession(mode=cqlMode)

table_name = 'electricity'

# For the real case, we take a 24h window
# Take the first 16h as the portion to embedd
# We do a step of every hour to get as many samples as possible
SLIDING_WINDOW_SIZE = 24
PORTION_TO_EMBED = 16
STEP = 1

"""load_sample_data
Load the sample data used to train vectorize and test the model
"""
@st.cache_data
def load_sample_data():
    df = pd.read_parquet('./data/electricity/train-00000-of-00001.parquet', engine='pyarrow')

    data = df.sort_index(axis=0, ascending=True)

    data["Date"] = pd.to_datetime(data["__index_level_0__"])
    data.set_index('Date', inplace=True)
    data = data.reset_index().rename(columns={'index': 'Date'})
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.drop(columns=['__index_level_0__'], axis=1)

    return data

data = load_sample_data()

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
test_data_size = len(test_data) - SLIDING_WINDOW_SIZE

# Load the Time2Vec model
d2v = Date2VecConvert(model_path="./d2v_model/d2v_98291_17.169918439404636.pth")

def get_step():
    return STEP

def get_sliding_window():
    return SLIDING_WINDOW_SIZE

def get_portion_to_embed():
    return PORTION_TO_EMBED

"""create_data_model
Data model used for electricity data
"""
def create_data_model():
    drop_table = f"DROP TABLE IF EXISTS {ASTRA_DB_KEYSPACE}.electricity"
    session.execute(drop_table)

    create_table = f"""
    CREATE TABLE IF NOT EXISTS {ASTRA_DB_KEYSPACE}.electricity (
    id text PRIMARY KEY,
    orig_timestamps vector<float, {SLIDING_WINDOW_SIZE}>,
    orig_demand VECTOR<float, {SLIDING_WINDOW_SIZE}>,
    orig_temperature VECTOR<float, {SLIDING_WINDOW_SIZE}>,
    embedding VECTOR<float, {1024}>,  
    );
    """
    session.execute(create_table)

    create_index = f"""
    CREATE CUSTOM INDEX IF NOT EXISTS demand_embedding_index 
    ON {ASTRA_DB_KEYSPACE}.electricity(embedding) 
    USING 'StorageAttachedIndex'
    WITH OPTIONS = {{ 'similarity_function': 'dot_product' }}
    ;
    """
    session.execute(create_index)

"""get_train_test_split
break the dataset into train and test data
"""
def get_train_test_split():
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    test_data_size = len(test_data) - SLIDING_WINDOW_SIZE

    return train_data, test_data, test_data_size, len(data)

"""get_windows
return a dataframe of windows for a dataset based on desired window size and step
"""
def get_windows(data):
    r = np.arange(len(data))
    s = r[::STEP]
    z = list(zip(s, s + SLIDING_WINDOW_SIZE))
    f = '{0[0]}:{0[1]}'.format
    g = lambda t: data.iloc[t[0]:t[1]]
    return pd.concat(map(g, z), keys=map(f, z))

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

"""create_windows_to_insert
Create the windows to be inserted in the database
"""
def create_windows_to_insert():
    wdf = get_windows(train_data)
    items_to_upload = []
    for window_i, window_df in wdf.groupby(level=0):
        if window_df.shape[0] == SLIDING_WINDOW_SIZE:
            window_df['Date'] = window_df['Date'].astype(np.int64)
            half_window_df = window_df.head(PORTION_TO_EMBED)
            embedding = get_embedding_for_window(half_window_df)
            if embedding:
                items_to_upload.append((window_df, embedding))
    return items_to_upload

def load_timeseries():
    create_data_model()
    items_to_upload = create_windows_to_insert
    logging.getLogger('cassandra').setLevel(logging.ERROR)

    prepared_insert = session.prepare(f"""
        INSERT INTO {ASTRA_DB_KEYSPACE}.electricity 
        (id, orig_timestamps, orig_demand, orig_temperature, embedding) 
        VALUES (?, ?, ?, ?, ?)
        """)

    print(f"Uploading {len(items_to_upload)} items.")

    nl = '\n'

    with st.spinner('Loading time series windows...'):
        for window_df, embeddings in items_to_upload:
            row_id = window_df.index[0][0] # 0:64 ~ identifier of the indices of the values
            timestamps = window_df['Date'].values.tolist()
            demands = window_df['Demand'].values.tolist()
            temperatures = window_df['Temperature'].values.tolist()
            embeddings = embeddings
            # wrapping in a loop to do naiive retries
            while True:
                try:
                    session.execute(prepared_insert, [row_id, timestamps, demands, temperatures, embeddings])
                    break
                except Exception as e:
                    print(e)
                    print(f'id was: {row_id} :: Vector sizes: {len(timestamps)} {len(demands)} {len(temperatures)} {len(embeddings)}')
                    break    



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
def get_plot_data(point):
    
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

