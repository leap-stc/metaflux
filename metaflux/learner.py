import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import learn2learn as l2l
import copy
from .model import Model
from .encoder import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def shuffle(tensors):
    batch_sz = tensors[0].shape[0]
    shuffled_index = torch.randperm(batch_sz)

    for i, tensor in enumerate(tensors):
        tensors[i] = tensors[i][shuffled_index]
    return tensors

class Learner():
    """
    Defining the base learner. Currently supports 'linear', 'lstm', and 'bilstm' 

    Params:
    -------
    input_size: int
        The expected input size to the model
    hidden_size: int
        The expected hidden size to the model
    model_type: str
        The base model type, currently supports ["mlp", "lstm", "bilstm"]
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        model_type: str, 
        fluxnet_train,
        fluxnet_test,
        update_lr,
        meta_lr,
        batch_size,
        max_meta_step,
        finetune_size,
        encoder_hidden_size: int = 32,
        with_context: bool = False,
        with_baseline: bool = False
    ) -> None:

        super(Learner, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model_type = model_type
        self.fluxnet_train = fluxnet_train
        self.fluxnet_test = fluxnet_test
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.batch_size = batch_size
        self.max_meta_step = max_meta_step
        self.finetune_size = finetune_size
        self.encoder_hidden_size = encoder_hidden_size
        self.with_context = with_context
        self.with_baseline = with_baseline
        self.encoder_input_size = next(iter(fluxnet_train))[0].shape[-1] - self.input_size
        self.loss = nn.MSELoss(reduction="mean")

        self.meta_loss_metric = dict()
        if self.with_baseline:
            self.base_loss_metric = dict()


    def train_meta(
        self, 
        runs, 
        epochs, 
        verbose=False
    ) -> None:
        """Main function to train metalearning algorithm"""

        for run in range(0,runs):
            self.model = Model(
                self.model_type,
                self.input_size,
                self.hidden_size,
                self.encoder_hidden_size,
                self.with_context
            ).to(device)
            self.maml = l2l.algorithms.MAML(self.model, lr=self.update_lr, first_order=False)
            opt = torch.optim.Adam(self.maml.parameters(), lr=self.meta_lr)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

            support_dl = DataLoader(self.fluxnet_train, batch_size=self.batch_size, shuffle=True)
            query_dl = DataLoader(self.fluxnet_test, batch_size=self.batch_size, shuffle=True)

            if self.with_context:
                self.encoder = Encoder(
                    input_size = self.encoder_input_size,
                    hidden_size = self.encoder_hidden_size
                ).double().to(device)
                
                encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.update_lr)
                encoder_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_opt, T_max=epochs)
            else:
                self.encoder, encoder_opt, encoder_schedule = None, None, None

            train_epoch, val_epoch = list(), list()
            
            for epoch in range(epochs):
                opt.zero_grad()
                if self.with_context:
                    encoder_opt.zero_grad()
                    
                train_error, val_error, outer_error = 0.0, 0.0, 0.0

                # Main loop
                with torch.backends.cudnn.flags(enabled=False):
                    learner = self.maml.clone().double()

                    for task, (support, query) in enumerate(zip(support_dl, query_dl)):
                        support_x, support_y = support
                        query_x, query_y = query
                        support_x, support_y = support_x.to(device), support_y.to(device)
                        query_x, query_y = query_x.to(device), query_y.to(device)

                        support_train_x, support_test_x, support_train_y, support_test_y = train_test_split(support_x, support_y, test_size=0.2, random_state=42)
                        query_train_x, query_test_x, query_train_y, query_test_y = train_test_split(query_x, query_y, test_size=0.2, random_state=42)
                        
                        # Meta-training
                        pred = self._get_pred(support_train_x, learner)
                        error = self.loss(pred, support_train_y)
                        learner.adapt(error)

                        # Meta-learning evaluation (no gradient step)
                        pred = self._get_pred(support_test_x, learner)
                        error = self.loss(pred, support_test_y)
                        train_error += error.item()
                        
                        # Update
                        pred = self._get_pred(query_train_x, learner)
                        error = self.loss(pred, query_train_y)
                        outer_error += error

                        # Adaptation evaluation (no gradient step)
                        pred = self._get_pred(query_test_x, learner)
                        error = self.loss(pred, query_test_y)
                        val_error += error.item()

                        del support_x, support_y, query_x, query_y
                        torch.cuda.empty_cache()

                        if task + 1 == self.max_meta_step:
                            train_epoch.append(train_error/(task + 1))
                            val_epoch.append(val_error/(task + 1))
                            break

                    # Parameter outer-loop update
                    outer_error.backward()
                    self._grad_step(task, opt, schedule, encoder_opt, encoder_schedule)
                
                if verbose and ((epoch % 50 == 0) or (epoch == (epochs - 1))):
                    print(f'Epoch: {epoch}, training loss: {train_epoch[epoch]}, validation loss: {val_epoch[epoch]}')

            self.meta_loss_metric.update({
                f"train_epoch_{run}": np.sqrt(np.array(train_epoch)),
                f"val_epoch_{run}": np.sqrt(np.array(val_epoch))
            })

        if self.with_baseline:
            self._train_base(runs, epochs, verbose)

    def _train_base(
        self, 
        runs, 
        epochs, 
        verbose=False
    ) -> None:
        """Only active when with_baseline parameter is True: train a baseline without the MAML algorithm"""

        for run in range(0,runs):
            self.base_model = Model(
                self.model_type,
                self.input_size,
                self.hidden_size,
                self.encoder_hidden_size,
                with_context=False
            ).double().to(device)
            opt = torch.optim.Adam(self.base_model.parameters(), lr=self.update_lr)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

            support_dl = DataLoader(self.fluxnet_train, batch_size=self.batch_size, shuffle=True)
            query_dl = DataLoader(self.fluxnet_test, batch_size=self.batch_size, shuffle=True)

            train_epoch, val_epoch = list(), list()
            
            for epoch in range(epochs):
                opt.zero_grad()
                train_error, val_error = 0.0, 0.0

                with torch.backends.cudnn.flags(enabled=False):
                    for task, (support, query) in enumerate(zip(support_dl, query_dl)):
                        support_x, support_y = support
                        query_x, query_y = query
                        support_x, support_y = support_x.to(device), support_y.to(device)
                        query_x, query_y = query_x.to(device), query_y.to(device)

                        support_train_x, support_test_x, support_train_y, support_test_y = train_test_split(support_x, support_y, test_size=0.2, random_state=42)
                        query_train_x, query_test_x, query_train_y, query_test_y = train_test_split(query_x, query_y, test_size=0.2, random_state=42)
                        
                        # Baseline learning + evaluation
                        pred = self.base_model(support_train_x[:,:,:self.input_size])
                        error = self.loss(pred, support_train_y)
                        error.backward()
                        opt.step()

                        ## Note: no gradient step
                        train_error += self.loss(self.base_model(support_test_x[:,:,:self.input_size]), 
                                                 support_test_y).item()

                        # Baseline adaptation + validation
                        pred = self.base_model(query_train_x[:,:,:self.input_size])
                        error = self.loss(pred, query_train_y)
                        error.backward()
                        opt.step()

                        ## Note: no gradient step
                        val_error += self.loss(self.base_model(query_test_x[:,:,:self.input_size]), 
                                               query_test_y).item()

                        schedule.step()

                        if task + 1 == self.max_meta_step:
                            train_epoch.append(train_error/(task + 1))
                            val_epoch.append(val_error/(task + 1))
                            break
                
                # No outer adaptation here
                if verbose and ((epoch % 50 == 0) or (epoch == (epochs - 1))):
                    print(f'Epoch: {epoch}, training loss: {train_epoch[epoch]}, validation loss: {val_epoch[epoch]}')

            self.base_loss_metric.update({
                f"train_epoch_{run}": np.sqrt(np.array(train_epoch)),
                f"val_epoch_{run}": np.sqrt(np.array(val_epoch))
            })

    
    def _get_pred(
        self, 
        x, 
        learner
    ) -> None:
        "Subroutine to perform prediction on input x"

        if self.with_context:
            encoding = self.encoder(x[:,:,self.input_size:])
            learner.module.update_encoding(encoding)

        return learner(x[:,:,:self.input_size])

    def _grad_step(
        self, 
        task,
        opt,
        schedule,
        encoder_opt,
        encoder_schedule
    ) -> None:
        "Subroutine to perform gradient step"

        for _, p in self.maml.named_parameters():
            p.grad.data.mul_(1.0/(task + 1))
        
        opt.step()
        schedule.step()
        
        if self.with_context:
            for _, p in self.encoder.named_parameters():
                p.grad.data.mul_(1.0/(task + 1))
            encoder_opt.step()
            encoder_schedule.step()