import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learner(nn.Module):
    """
    Defining the base learner. Currently supports 'linear', 'lstm', and 'bilstm' 

    Params:
    -------
    config: dict
        A dictionary containing the specification of models
    input_size: int
        The expected input size to the model
    hidden_size: int
        The expected hidden size to the model
    """
    def __init__(self, config, input_size, hidden_size):
        super(Learner, self).__init__()
        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #it contains all tensors that need to be optimized
        self.vars = nn.ParameterList()
        
        
        for i, (name, param) in enumerate(self.config):

            if name == 'linear':
                w = nn.Parameter(torch.ones(*param, dtype=torch.float64, device=device))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0], dtype=torch.float64, device=device)))

            elif name == 'lstm':
                self.lstm = torch.nn.LSTM(input_size = self.input_size, 
                                     hidden_size = self.hidden_size,
                                     bidirectional=False,
                                     num_layers=1, 
                                     batch_first=True)
                # ih_layer
                w1 = nn.Parameter(torch.ones(param[0:2], dtype=torch.float64, device=device))
                torch.nn.init.kaiming_normal_(w1)
                self.vars.append(w1)
                self.vars.append(nn.Parameter(torch.zeros(param[0], dtype=torch.float64, device=device)))
                # hh_layer
                w2 = nn.Parameter(torch.ones((param[0], param[2]), dtype=torch.float64, device=device))
                torch.nn.init.kaiming_normal_(w2)
                self.vars.append(w2)
                self.vars.append(nn.Parameter(torch.zeros(param[0], dtype=torch.float64, device=device)))
            
            elif name == 'bilstm':
                self.lstm = torch.nn.LSTM(input_size = self.input_size, 
                                     hidden_size = self.hidden_size,
                                     bidirectional=True,
                                     num_layers=1, 
                                     batch_first=True)
                # ih_layer
                w1 = nn.Parameter(torch.ones(param[0:2], dtype=torch.float64, device=device))
                torch.nn.init.kaiming_normal_(w1)
                self.vars.append(w1)
                self.vars.append(nn.Parameter(torch.zeros(param[0], dtype=torch.float64, device=device)))
                # hh_layer
                w2 = nn.Parameter(torch.ones((param[0], param[2]), dtype=torch.float64, device=device))
                torch.nn.init.kaiming_normal_(w2)
                self.vars.append(w2)
                self.vars.append(nn.Parameter(torch.zeros(param[0], dtype=torch.float64, device=device)))
                
                # ih_layer_reverse
                w3 = nn.Parameter(torch.ones(param[0:2], dtype=torch.float64, device=device))
                torch.nn.init.kaiming_normal_(w3)
                self.vars.append(w3)
                self.vars.append(nn.Parameter(torch.zeros(param[0], dtype=torch.float64, device=device)))
                # hh_layer_reverse
                w4 = nn.Parameter(torch.ones((param[0], param[2]), dtype=torch.float64, device=device))
                torch.nn.init.kaiming_normal_(w4)
                self.vars.append(w4)
                self.vars.append(nn.Parameter(torch.zeros(param[0], dtype=torch.float64, device=device)))

                
            elif name in ['tanh', 'relu', 'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError
        
            
    def forward(self, x, vars=None, is_train=True):
        """
        :param x
        :param vars
        :return x
        """
            
        if is_train and vars != None:
            # assign new parameters to the network if it is training
            for i, param in enumerate(self.vars):
                self.vars[i] = vars[i]
            
            
        idx = 0

        for name, param in self.config:
            if name == "lstm":
                self.lstm.weight_ih_l0, self.lstm.bias_ih_l0 = self.vars[idx], self.vars[idx + 1]
                self.lstm.weight_hh_l0, self.lstm.bias_hh_l0 = self.vars[idx + 2], self.vars[idx + 3]
                
                x, (_, _) = self.lstm(x)
                idx += 4
                
            elif name == "bilstm":
                self.lstm.weight_ih_l0, self.lstm.bias_ih_l0 = self.vars[idx], self.vars[idx + 1]
                self.lstm.weight_hh_l0, self.lstm.bias_hh_l0 = self.vars[idx + 2], self.vars[idx + 3]
                self.lstm.weight_ih_l0_reverse, self.lstm.bias_ih_l0_reverse = self.vars[idx + 4], self.vars[idx + 5]
                self.lstm.weight_hh_l0_reverse, self.lstm.bias_hh_l0_reverse = self.vars[idx + 6], self.vars[idx + 7]
                
                x, (_, _) = self.lstm(x)
                idx += 8
            
            elif name == "linear":
                w, b = self.vars[idx], self.vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            
            elif name == "leakyrelu":
                x = F.leaky_relu(x, negative_slope=param[0])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(self.vars)
        
        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars == None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


class BaseMeta(nn.Module):
    """
    Defining the base learner (for use without meta-learning)
    TODO: use config to define model structure as meta-learning approach above

    Params:
    -------
    config: dict
        A dictionary containing the specification of models
    input_size: int
        The expected input size to the model
    hidden_size: int
        The expected hidden size to the model
    """

    def __init__(self, input_size, hidden_size, arch, neg_slope=0.01):
        super(BaseMeta, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.arch = arch
        self.neg_slope = neg_slope
        
        if self.arch == "lstm":
            self.lstm = torch.nn.LSTM(input_size = self.input_size, 
                                        hidden_size = self.hidden_size,
                                        bidirectional=False,
                                        num_layers=2, 
                                        batch_first=True)
                                        
            self.fc_out = torch.nn.Linear(self.hidden_size, 1) #for unidirectional
            
        elif self.arch == "bilstm":
            self.lstm = torch.nn.LSTM(input_size = self.input_size, 
                            hidden_size = self.hidden_size,
                            bidirectional=True,
                            num_layers=2, 
                            batch_first=True)

            self.fc_out = torch.nn.Linear(2*self.hidden_size, 1) #for bidirectional
            
        else:
            self.fc_in = torch.nn.Linear(self.input_size, self.hidden_size)
            self.fc_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.fc_out = torch.nn.Linear(self.hidden_size, 1)
        
    def forward(self, x):
        if self.arch == "lstm":
            x, (_,_) = self.lstm(x)
            x = F.leaky_relu(x, negative_slope=self.neg_slope)
            x = self.fc_out(x)

        elif self.arch == "bilstm":
            x, (_,_) = self.lstm(x)
            x = F.leaky_relu(x, negative_slope=self.neg_slope)
            x = self.fc_out(x)
        
        else:
            x = self.fc_in(x)
            x = F.leaky_relu(x, negative_slope=self.neg_slope)
            x = self.fc_1(x)
            x = F.leaky_relu(x, negative_slope=self.neg_slope)
            x = self.fc_out(x)
        
        return x