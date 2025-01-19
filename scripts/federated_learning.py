{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
\
class FederatedLearning:\
    def __init__(self, clients, global_model, epochs=10, algorithm="FedAvg", mu=0.01):\
        """\
        Initialize Federated Learning instance.\
        Args:\
            clients (list): List of clients participating in training.\
            global_model (torch.nn.Module): Global model instance.\
            epochs (int): Number of local epochs for training.\
            algorithm (str): Federated learning algorithm ('FedAvg' or 'FedProx').\
            mu (float): Proximal term weight for FedProx.\
        """\
        self.clients = clients\
        self.global_model = global_model\
        self.epochs = epochs\
        self.algorithm = algorithm\
        self.mu = mu\
\
    def aggregate_weights(self, client_weights, client_sizes):\
        """\
        Aggregate client weights using weighted average.\
        Args:\
            client_weights (list): List of model weights from clients.\
            client_sizes (list): Number of samples per client.\
        Returns:\
            Aggregated weights.\
        """\
        total_samples = sum(client_sizes)\
        aggregated_weights = \{\
            key: sum(client_weights[i][key] * client_sizes[i] / total_samples for i in range(len(client_weights)))\
            for key in client_weights[0].keys()\
        \}\
        return aggregated_weights\
\
    def train(self):\
        """\
        Perform federated learning training rounds.\
        """\
        for round_num in range(1, self.epochs + 1):\
            print(f"Round \{round_num\} started.")\
            \
            client_weights = []\
            client_sizes = []\
\
            # Local training for each client\
            for client in self.clients:\
                local_model, num_samples = client.train(self.global_model, self.epochs, self.algorithm, self.mu)\
                client_weights.append(local_model.state_dict())\
                client_sizes.append(num_samples)\
\
            # Aggregate weights on the server\
            aggregated_weights = self.aggregate_weights(client_weights, client_sizes)\
            self.global_model.load_state_dict(aggregated_weights)\
            \
            print(f"Round \{round_num\} completed.")\
\
        return self.global_model\
\
class Client:\
    def __init__(self, model, data_loader, optimizer, loss_fn, device="cpu"):\
        """\
        Initialize Client instance.\
        Args:\
            model (torch.nn.Module): Local model instance.\
            data_loader (DataLoader): DataLoader for local training data.\
            optimizer (torch.optim.Optimizer): Optimizer for local training.\
            loss_fn: Loss function for training.\
            device (str): Device to run the training (e.g., 'cpu' or 'cuda').\
        """\
        self.model = model\
        self.data_loader = data_loader\
        self.optimizer = optimizer\
        self.loss_fn = loss_fn\
        self.device = device\
\
    def train(self, global_model, epochs, algorithm="FedAvg", mu=0.01):\
        """\
        Train the local model.\
        Args:\
            global_model (torch.nn.Module): Global model for initialization.\
            epochs (int): Number of local epochs.\
            algorithm (str): Federated learning algorithm ('FedAvg' or 'FedProx').\
            mu (float): Proximal term weight for FedProx.\
        Returns:\
            Trained local model and number of samples.\
        """\
        self.model.load_state_dict(global_model.state_dict())\
        self.model.to(self.device)\
        self.model.train()\
\
        for epoch in range(epochs):\
            for batch_idx, (data, target) in enumerate(self.data_loader):\
                data, target = data.to(self.device), target.to(self.device)\
                self.optimizer.zero_grad()\
                \
                output = self.model(data)\
                loss = self.loss_fn(output, target)\
                \
                # Apply FedProx regularization\
                if algorithm == "FedProx":\
                    fed_prox_loss = 0.0\
                    for param, global_param in zip(self.model.parameters(), global_model.parameters()):\
                        fed_prox_loss += ((param - global_param) ** 2).sum()\
                    loss += (mu / 2) * fed_prox_loss\
                \
                loss.backward()\
                self.optimizer.step()\
\
        num_samples = len(self.data_loader.dataset)\
        return self.model, num_samples}