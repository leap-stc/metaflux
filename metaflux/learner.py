import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import learn2learn as l2l
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
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        model_type: str, 
        fluxnet_base,
        fluxnet_target,
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
        self.fluxnet_base = fluxnet_base
        self.fluxnet_target = fluxnet_target
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.batch_size = batch_size
        self.max_meta_step = max_meta_step
        self.finetune_size = finetune_size
        self.encoder_hidden_size = encoder_hidden_size
        self.with_context = with_context
        self.with_baseline = with_baseline
        self.encoder_input_size = next(iter(fluxnet_base))[0].shape[-1] - self.input_size
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

        # Each run --> new model
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

            # processing the data
            base_train_sz = int(len(self.fluxnet_base) * (1 - self.finetune_size))
            target_train_sz = int(len(self.fluxnet_target) * (1 - self.finetune_size))

            # split both datasets to meta-train and meta-test
            base_train, base_test = torch.utils.data.random_split(self.fluxnet_base, [base_train_sz, len(self.fluxnet_base) - base_train_sz])
            target_train, target_test = torch.utils.data.random_split(self.fluxnet_target, [target_train_sz, len(self.fluxnet_target) - target_train_sz])

            base_train_dl = DataLoader(base_train, batch_size=self.batch_size, shuffle=True)
            base_test_dl = DataLoader(base_test, batch_size=self.batch_size, shuffle=True)
            target_train_dl = DataLoader(target_train, batch_size=self.batch_size, shuffle=True)
            target_test_dl = DataLoader(target_test, batch_size=self.batch_size, shuffle=True)

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

                # Get k-shot batches
                base_train_k = self._get_k_shot(base_train_dl, self.max_meta_step)
                base_test_k = self._get_k_shot(base_test_dl, self.max_meta_step)
                target_train_k = self._get_k_shot(target_train_dl, self.max_meta_step)
                target_test_k = self._get_k_shot(target_test_dl, self.max_meta_step)

                # Main loop
                with torch.backends.cudnn.flags(enabled=False):
                    learner = self.maml.clone().double()

                    # Propose phi using base sets
                    for task, (b_train, b_test) in enumerate(zip(base_train_k, base_test_k)):
                        b_train_x, b_train_y = b_train
                        b_test_x, b_test_y = b_test
                        b_train_x, b_train_y = b_train_x.to(device), b_train_y.to(device)
                        b_test_x, b_test_y = b_test_x.to(device), b_test_y.to(device)

                        ## Inner-loop to propose phi using meta-training dataset
                        pred = self._get_pred(b_train_x, learner)
                        error = self.loss(pred, b_train_y)
                        learner.adapt(error)

                        # Inner-loop evaluation (no gradient step) using meta-testing dataset
                        pred = self._get_pred(b_test_x, learner)
                        error = self.loss(pred, b_test_y)
                        train_error += error.item()
                    
                    train_epoch.append(train_error/(task + 1))

                    # Outer-loop using target sets
                    for task, (t_train, t_test) in enumerate(zip(target_train_k, target_test_k)):
                        t_train_x, t_train_y = t_train
                        t_test_x, t_test_y = t_test
                        t_train_x, t_train_y = t_train_x.to(device), t_train_y.to(device)
                        t_test_x, t_test_y = t_test_x.to(device), t_test_y.to(device)

                        ## accumulate inner-loop gradients given proposed phi
                        pred = self._get_pred(t_train_x, learner)
                        error = self.loss(pred, t_train_y)
                        outer_error += error

                        # Adaptation evaluation (no gradient step) using meta-testing dataset
                        pred = self._get_pred(t_test_x, learner)
                        error = self.loss(pred, t_test_y)
                        val_error += error.item()

                    val_epoch.append(val_error/(task + 1))

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

            # processing the data
            base_train_sz = int(len(self.fluxnet_base) * (1 - self.finetune_size))
            target_train_sz = int(len(self.fluxnet_target) * (1 - self.finetune_size))

            # split both datasets to train and test
            base_train, base_test = torch.utils.data.random_split(self.fluxnet_base, [base_train_sz, len(self.fluxnet_base) - base_train_sz])
            target_train, target_test = torch.utils.data.random_split(self.fluxnet_target, [target_train_sz, len(self.fluxnet_target) - target_train_sz])

            base_train_dl = DataLoader(base_train, batch_size=self.batch_size, shuffle=True)
            base_test_dl = DataLoader(base_test, batch_size=self.batch_size, shuffle=True)
            target_train_dl = DataLoader(target_train, batch_size=self.batch_size, shuffle=True)
            target_test_dl = DataLoader(target_test, batch_size=self.batch_size, shuffle=True)

            train_epoch, val_epoch = list(), list()
            
            for epoch in range(epochs):
                opt.zero_grad()
                train_error, val_error = 0.0, 0.0

                # Get k-shot batches
                base_train_k = self._get_k_shot(base_train_dl, self.max_meta_step)
                base_test_k = self._get_k_shot(base_test_dl, self.max_meta_step)
                target_train_k = self._get_k_shot(target_train_dl, self.max_meta_step)
                target_test_k = self._get_k_shot(target_test_dl, self.max_meta_step)

                with torch.backends.cudnn.flags(enabled=False):
                    # Baseline learning + evaluation
                    for task, (b_train, b_test) in enumerate(zip(base_train_k, base_test_k)):
                        b_train_x, b_train_y = b_train
                        b_test_x, b_test_y = b_test
                        b_train_x, b_train_y = b_train_x.to(device), b_train_y.to(device)
                        b_test_x, b_test_y = b_test_x.to(device), b_test_y.to(device)

                        pred = self.base_model(b_train_x[:,:,:self.input_size])
                        error = self.loss(pred, b_train_y)
                        error.backward()
                        opt.step()

                        ## Note: no gradient step
                        train_error += self.loss(self.base_model(b_test_x[:,:,:self.input_size]), 
                                                 b_test_y).item()
                        
                    train_epoch.append(train_error/(task + 1))

                    # Baseline-equivalent to meta-adaptation + validation
                    for task, (t_train, t_test) in enumerate(zip(target_train_k, target_test_k)):
                        t_train_x, t_train_y = t_train
                        t_test_x, t_test_y = t_test
                        t_train_x, t_train_y = t_train_x.to(device), t_train_y.to(device)
                        t_test_x, t_test_y = t_test_x.to(device), t_test_y.to(device)

                        pred = self.base_model(t_train_x[:,:,:self.input_size])
                        error = self.loss(pred, t_train_y)
                        error.backward()
                        opt.step()

                        ## Note: no gradient step
                        val_error += self.loss(self.base_model(t_test_x[:,:,:self.input_size]), 
                                               t_test_y).item()

                        schedule.step()

                    val_epoch.append(val_error/(task + 1))
                
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
        
        # normalize loss with the number of task
        for _, p in self.maml.named_parameters():
            p.grad.data.mul_(1.0/(task + 1))
        
        opt.step()
        schedule.step()
        
        if self.with_context:
            for _, p in self.encoder.named_parameters():
                p.grad.data.mul_(1.0/(task + 1))
            encoder_opt.step()
            encoder_schedule.step()

    def _get_k_shot(
            self,
            dataloader,
            k
    ):
        "Get k-shot samples from each batch"
        selected_dataloader = itertools.islice(iter(dataloader), 0, min(len(dataloader), k))
        return selected_dataloader