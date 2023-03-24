import torch
from torch import nn as nn

class Model(nn.Module):
    """
    Defining the base learners. Currently supports one of [mlp, lstm, bilstm]

    Params:
    -------
    model_type: str
        The base architecture of the model, one of [mlp, lstm, bilstm]
    input_size: int
        The expected input size to the model
    hidden_size: int
        The expected hidden size to the model
    encoder_hidden_size: int
        Size of the hidden layer for the encoder (only if with_context==True)
    with_context: bool
        Flag indicating the usage of context encoder
    """
    
    def __init__(self, model_type, input_size, hidden_size, encoder_hidden_size, with_context):
        super(Model, self).__init__()
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.with_context = with_context

        if self.model_type == "mlp":
            self.linear_in = nn.Linear(self.input_size, self.hidden_size)
            self.linear_h = nn.Linear(self.hidden_size, self.hidden_size) 

        elif self.model_type == 'lstm':
            self.lstm = nn.LSTM(input_size = self.input_size, 
                                    hidden_size = self.hidden_size,
                                    bidirectional=False,
                                    num_layers=1, 
                                    batch_first=True)
            self.linear_h = nn.Linear(self.hidden_size, self.hidden_size) 
        
        elif self.model_type == 'bilstm':
            self.lstm = nn.LSTM(input_size = self.input_size, 
                                    hidden_size = self.hidden_size,
                                    bidirectional=True,
                                    num_layers=1, 
                                    batch_first=True)

            self.linear_h = nn.Linear(self.hidden_size * 2, self.hidden_size)
          
        self.leaky_relu = nn.LeakyReLU(0.01)
        
        if self.with_context:
            self.linear_out = nn.Linear(self.hidden_size + self.encoder_hidden_size, 1)
        else:
            self.linear_out = nn.Linear(self.hidden_size, 1)
    
    def update_encoding(self, encoding) -> None:
        self.encoding = encoding
            
    def forward(self, x):
        """
        :param x
        :return x
        """
        if self.model_type == "mlp":
            x = self.linear_in(x)
            x = self.leaky_relu(x)
            x = self.linear_h(x)
            x = self.leaky_relu(x)

        else:
            x, (_, _) = self.lstm(x)
            x = self.linear_h(x)
            x = self.leaky_relu(x)

        if self.with_context:
            assert self.encoding != None
            x = torch.cat((x, self.encoding), axis=2)

        x = self.linear_out(x)

        return x.reshape((x.shape[0], x.shape[1]))