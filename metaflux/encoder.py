import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Defining the Encoder

    Params:
    -------
    input_size: int
        The expected input size to the model
    hidden_size: int
        The expected hidden size to the model
    """
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_in = nn.Linear(self.input_size, self.hidden_size)
        self.linear_h = nn.Linear(self.hidden_size, self.hidden_size) 
        self.linear_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.leaky_relu = nn.LeakyReLU(0.01)
            
    def forward(self, x):
        """
        :param x
        :return x
        """
        x = self.linear_in(x)
        x = self.leaky_relu(x)
        x = self.linear_h(x)
        x = self.leaky_relu(x)
        x = self.linear_out(x)
        
        return x