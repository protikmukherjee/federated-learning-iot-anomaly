{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import torch\
\
class FederatedAveraging:\
    """\
    Implements the Federated Averaging (FedAvg) algorithm for model aggregation.\
    """\
    @staticmethod\
    def aggregate(global_model, client_models, client_data_sizes):\
        """\
        Aggregates client model weights using Federated Averaging.\
\
        Args:\
            global_model: The global PyTorch model.\
            client_models: List of client models (PyTorch state_dict).\
            client_data_sizes: List of data sizes for each client.\
\
        Returns:\
            Updated global model with aggregated weights.\
        """\
        total_data_size = sum(client_data_sizes)\
        global_state_dict = global_model.state_dict()\
\
        # Initialize the global model's parameters to zero\
        for key in global_state_dict:\
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])\
\
        # Aggregate weights from each client\
        for client_idx, client_model in enumerate(client_models):\
            client_state_dict = client_model.state_dict()\
            weight = client_data_sizes[client_idx] / total_data_size\
            for key in client_state_dict:\
                global_state_dict[key] += client_state_dict[key] * weight\
\
        # Update the global model's state_dict\
        global_model.load_state_dict(global_state_dict)\
        return global_model\
\
\
class FederatedProximal:\
    """\
    Implements the Federated Proximal (FedProx) algorithm for model aggregation.\
    """\
    def __init__(self, mu=0.01):\
        """\
        Args:\
            mu: Regularization parameter for the proximal term.\
        """\
        self.mu = mu\
\
    def aggregate(self, global_model, client_models, client_data_sizes):\
        """\
        Aggregates client model weights using Federated Proximal.\
\
        Args:\
            global_model: The global PyTorch model.\
            client_models: List of client models (PyTorch state_dict).\
            client_data_sizes: List of data sizes for each client.\
\
        Returns:\
            Updated global model with aggregated weights.\
        """\
        total_data_size = sum(client_data_sizes)\
        global_state_dict = global_model.state_dict()\
\
        # Initialize the global model's parameters to zero\
        for key in global_state_dict:\
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])\
\
        # Aggregate weights with FedProx's proximal term\
        for client_idx, client_model in enumerate(client_models):\
            client_state_dict = client_model.state_dict()\
            weight = client_data_sizes[client_idx] / total_data_size\
            for key in client_state_dict:\
                diff = client_state_dict[key] - global_state_dict[key]\
                global_state_dict[key] += client_state_dict[key] * weight - self.mu * diff\
\
        # Update the global model's state_dict\
        global_model.load_state_dict(global_state_dict)\
        return global_model}