{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import torch\
import torch.nn as nn\
\
class Autoencoder(nn.Module):\
    def __init__(self, input_dim, latent_dim, dropout=0.2):\
        super(Autoencoder, self).__init__()\
        # Encoder\
        self.encoder = nn.Sequential(\
            nn.Linear(input_dim, 128),\
            nn.ReLU(),\
            nn.Dropout(dropout),\
            nn.Linear(128, latent_dim),\
            nn.ReLU()\
        )\
        # Decoder\
        self.decoder = nn.Sequential(\
            nn.Linear(latent_dim, 128),\
            nn.ReLU(),\
            nn.Dropout(dropout),\
            nn.Linear(128, input_dim),\
            nn.Sigmoid()  # Assuming normalized data; modify if not\
        )\
\
    def forward(self, x):\
        """\
        Args:\
            x: Input tensor of shape (batch_size, input_dim)\
        Returns:\
            Reconstructed input of the same shape as x\
        """\
        latent = self.encoder(x)\
        reconstructed = self.decoder(latent)\
        return reconstructed, latent}