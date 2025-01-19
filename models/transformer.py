{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import torch\
import torch.nn as nn\
import torch.nn.functional as F\
\
class TransformerEncoder(nn.Module):\
    def __init__(self, input_dim, model_dim, num_heads, ff_hidden_dim, num_layers, dropout=0.1):\
        super(TransformerEncoder, self).__init__()\
        self.model_dim = model_dim\
        self.encoder_layers = nn.TransformerEncoderLayer(\
            d_model=model_dim,\
            nhead=num_heads,\
            dim_feedforward=ff_hidden_dim,\
            dropout=dropout,\
            activation='relu'\
        )\
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)\
        self.input_projection = nn.Linear(input_dim, model_dim)\
        self.positional_encoding = PositionalEncoding(model_dim, dropout)\
\
    def forward(self, x):\
        """\
        Args:\
            x: Input tensor of shape (batch_size, seq_len, input_dim)\
        Returns:\
            Output tensor of shape (batch_size, seq_len, model_dim)\
        """\
        # Add positional encoding to input projections\
        x = self.input_projection(x)\
        x = self.positional_encoding(x)\
        x = self.transformer_encoder(x)\
        return x\
\
\
class PositionalEncoding(nn.Module):\
    def __init__(self, model_dim, dropout=0.1, max_len=5000):\
        super(PositionalEncoding, self).__init__()\
        self.dropout = nn.Dropout(p=dropout)\
\
        position = torch.arange(0, max_len).unsqueeze(1)\
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-torch.log(torch.tensor(10000.0)) / model_dim))\
        pe = torch.zeros(max_len, model_dim)\
        pe[:, 0::2] = torch.sin(position * div_term)\
        pe[:, 1::2] = torch.cos(position * div_term)\
        self.register_buffer('pe', pe)\
\
    def forward(self, x):\
        """\
        Add positional encoding to input tensor.\
        """\
        x = x + self.pe[:x.size(1), :].unsqueeze(0)\
        return self.dropout(x)}