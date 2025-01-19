{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import torch\
import torch.nn as nn\
import torch.optim as optim\
from federated_models import FederatedAveraging, FederatedProximal\
\
def train_local_model(model, dataloader, optimizer, criterion, device, epochs=1):\
    """\
    Trains the local model on a client's data.\
\
    Args:\
        model: PyTorch model to train.\
        dataloader: DataLoader for the client's local dataset.\
        optimizer: Optimizer for training.\
        criterion: Loss function.\
        device: Device to run the training on.\
        epochs: Number of training epochs.\
\
    Returns:\
        Trained model.\
    """\
    model.train()\
    model.to(device)\
    for _ in range(epochs):\
        for data, labels in dataloader:\
            data, labels = data.to(device), labels.to(device)\
            optimizer.zero_grad()\
            outputs = model(data)\
            loss = criterion(outputs, labels)\
            loss.backward()\
            optimizer.step()\
    return model\
\
\
def evaluate_model(model, dataloader, criterion, device):\
    """\
    Evaluates the model on a given dataset.\
\
    Args:\
        model: PyTorch model to evaluate.\
        dataloader: DataLoader for the evaluation dataset.\
        criterion: Loss function.\
        device: Device to run the evaluation on.\
\
    Returns:\
        accuracy, loss: Tuple of accuracy and loss on the evaluation dataset.\
    """\
    model.eval()\
    model.to(device)\
    total, correct = 0, 0\
    total_loss = 0.0\
    with torch.no_grad():\
        for data, labels in dataloader:\
            data, labels = data.to(device), labels.to(device)\
            outputs = model(data)\
            loss = criterion(outputs, labels)\
            total_loss += loss.item()\
            _, predicted = torch.max(outputs, 1)\
            correct += (predicted == labels).sum().item()\
            total += labels.size(0)\
    accuracy = 100 * correct / total\
    avg_loss = total_loss / len(dataloader)\
    return accuracy, avg_loss\
\
\
def federated_train(\
    global_model, client_data_loaders, global_epochs, local_epochs, criterion, device, strategy="fedavg", mu=0.01\
):\
    """\
    Performs federated training using the FedAvg or FedProx algorithm.\
\
    Args:\
        global_model: PyTorch model to act as the global model.\
        client_data_loaders: List of DataLoaders for each client's local data.\
        global_epochs: Number of communication rounds.\
        local_epochs: Number of epochs for local training.\
        criterion: Loss function.\
        device: Device to run training on.\
        strategy: Either "fedavg" or "fedprox" for the training strategy.\
        mu: Proximal term regularization parameter for FedProx.\
\
    Returns:\
        Trained global model.\
    """\
    if strategy == "fedavg":\
        aggregator = FederatedAveraging()\
    elif strategy == "fedprox":\
        aggregator = FederatedProximal(mu=mu)\
    else:\
        raise ValueError("Invalid strategy. Choose either 'fedavg' or 'fedprox'.")\
\
    for global_epoch in range(global_epochs):\
        client_models = []\
        client_data_sizes = []\
        print(f"Global Epoch \{global_epoch+1\}/\{global_epochs\}")\
\
        # Train on each client\
        for client_loader in client_data_loaders:\
            local_model = type(global_model)()  # Initialize a new instance of the model\
            local_model.load_state_dict(global_model.state_dict())\
            optimizer = optim.Adam(local_model.parameters(), lr=0.001)\
            local_model = train_local_model(local_model, client_loader, optimizer, criterion, device, local_epochs)\
            client_models.append(local_model)\
            client_data_sizes.append(len(client_loader.dataset))\
\
        # Aggregate models\
        global_model = aggregator.aggregate(global_model, client_models, client_data_sizes)\
\
    return global_model}