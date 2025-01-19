{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
import pandas as pd\
from sklearn.model_selection import train_test_split\
\
def load_dataset(filepath):\
    """\
    Load dataset from a CSV file.\
    Args:\
        filepath (str): Path to the dataset CSV file.\
    Returns:\
        pandas.DataFrame: Loaded dataset.\
    """\
    return pd.read_csv(filepath)\
\
def split_data_iid(data, target_column, num_clients):\
    """\
    Split data into IID subsets for federated learning.\
    Args:\
        data (pd.DataFrame): Full dataset.\
        target_column (str): Column name of the target variable.\
        num_clients (int): Number of clients.\
    Returns:\
        List of (X, y) pairs for each client.\
    """\
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle data\
    client_data = np.array_split(data, num_clients)\
    splits = [\
        (client.drop(columns=[target_column]).values, client[target_column].values)\
        for client in client_data\
    ]\
    return splits\
\
def split_data_noniid(data, target_column, num_clients, num_shards):\
    """\
    Split data into non-IID subsets for federated learning.\
    Args:\
        data (pd.DataFrame): Full dataset.\
        target_column (str): Column name of the target variable.\
        num_clients (int): Number of clients.\
        num_shards (int): Number of shards for non-IID split.\
    Returns:\
        List of (X, y) pairs for each client.\
    """\
    data = data.sort_values(by=target_column).reset_index(drop=True)\
    shards = np.array_split(data, num_shards)\
    client_data = [pd.concat(shards[i::num_clients]) for i in range(num_clients)]\
    splits = [\
        (client.drop(columns=[target_column]).values, client[target_column].values)\
        for client in client_data\
    ]\
    return splits\
\
def preprocess_data(data, feature_columns, target_column, test_size=0.2):\
    """\
    Preprocess data for training and evaluation.\
    Args:\
        data (pd.DataFrame): Full dataset.\
        feature_columns (list): List of feature column names.\
        target_column (str): Column name of the target variable.\
        test_size (float): Proportion of data for testing.\
    Returns:\
        X_train, X_test, y_train, y_test: Preprocessed train and test splits.\
    """\
    X = data[feature_columns]\
    y = data[target_column]\
    return train_test_split(X, y, test_size=test_size, random_state=42)}