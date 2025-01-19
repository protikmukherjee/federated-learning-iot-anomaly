{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import torch\
import torch.nn as nn\
import torch.optim as optim\
from torch.utils.data import DataLoader, TensorDataset\
from data_preprocessing import load_dataset, split_data_iid, split_data_noniid\
from transformer import TransformerModel\
from autoencoder import Autoencoder\
from federated_learning import FederatedLearning, Client\
\
# Hyperparameters\
NUM_CLIENTS = 10\
EPOCHS = 10\
ALGORITHM = "FedProx"  # Use "FedAvg" or "FedProx"\
MU = 0.01  # Regularization for FedProx\
LEARNING_RATE = 0.001\
BATCH_SIZE = 32\
\
# Load dataset\
data = load_dataset("dataset.csv")\
feature_columns = data.columns[:-1]\
target_column = data.columns[-1]\
\
# Create IID and Non-IID splits\
iid_splits = split_data_iid(data, target_column, NUM_CLIENTS)\
noniid_splits = split_data_noniid(data, target_column, NUM_CLIENTS, num_shards=20)\
\
# Create global model\
global_model = TransformerModel(input_dim=len(feature_columns), output_dim=1)\
\
# Initialize clients\
clients = []\
for X, y in iid_splits:  # Replace `iid_splits` with `noniid_splits` for non-IID\
    X_tensor = torch.tensor(X, dtype=torch.float32)\
    y_tensor = torch.tensor(y, dtype=torch.float32)\
    dataset = TensorDataset(X_tensor, y_tensor)\
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)\
\
    local_model = TransformerModel(input_dim=len(feature_columns), output_dim=1)\
    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)\
    loss_fn = nn.BCEWithLogitsLoss()\
\
    clients.append(Client(local_model, data_loader, optimizer, loss_fn))\
\
# Train using Federated Learning\
fl = FederatedLearning(clients, global_model, epochs=EPOCHS, algorithm=ALGORITHM, mu=MU)\
trained_global_model = fl.train()\
\
# Evaluate global model\
X_test = torch.tensor(data[feature_columns].values, dtype=torch.float32)\
y_test = torch.tensor(data[target_column].values, dtype=torch.float32)\
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)\
\
global_model.eval()\
with torch.no_grad():\
    predictions = []\
    for X_batch, _ in test_loader:\
        predictions.extend(global_model(X_batch).numpy())\
\
print("Global Model Evaluation Completed.")}