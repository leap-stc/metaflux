"""Legacy code: abstracted under Learner class as train() functionalities
"""

import metaflux
import learn2learn as l2l
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import copy

model_type= "mlp"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = nn.MSELoss(reduction="mean")

def train_meta(runs, hyper_args, fluxnet_train, fluxnet_test, with_context=True, for_inference=False, verbose=False):
    def _get_pred(x, learner, with_context):
        # latent = learner(x[:,:,:hyper_args["input_size"]])
        "Subroutine to perform prediction on input x"
        if with_context:
            encoding = encoder(x[:,:,hyper_args["input_size"]:])
            learner.module.update_encoding(encoding)

        return learner(x[:,:,:hyper_args["input_size"]])

    def _grad_step(with_context):
        for n, p in maml.named_parameters():
            p.grad.data.mul_(1.0/(step + 1))
        
        opt.step()
        schedule.step()
        
        if with_context:
            for n, p in encoder.named_parameters():
                p.grad.data.mul_(1.0/(step + 1))
            encoder_opt.step()
            encoder_schedule.step()
        
        return None

    meta_loss_metric = dict()
    encoder_input_size = next(iter(fluxnet_train))[0].shape[-1] - hyper_args["input_size"]

    for run in range(0,runs):
        model = metaflux.learner.Learner(
            input_size=hyper_args["input_size"], 
            hidden_size=hyper_args["hidden_size"], 
            model_type=model_type,
            encoder_hidden_size=hyper_args["encoder_hidden_size"],
            with_context=with_context
        ).to(device)
        maml = l2l.algorithms.MAML(model, lr=hyper_args["update_lr"], first_order=False)
        opt = torch.optim.Adam(maml.parameters(), lr=hyper_args["meta_lr"])

        if with_context:
            encoder = metaflux.encoder.Encoder(
                input_size = encoder_input_size,
                hidden_size = hyper_args["encoder_hidden_size"]
            ).double().to(device)
            
            encoder_opt = torch.optim.Adam(encoder.parameters(), lr=hyper_args["update_lr"])
            encoder_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_opt, T_max=hyper_args["epoch"])
        else:
            encoder = None

        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hyper_args["epoch"])
        train_epoch, val_epoch = list(), list()
        
        for epoch in range(hyper_args["epoch"]):
            opt.zero_grad()
            if with_context:
                encoder_opt.zero_grad()
                
            train_error, val_error = 0.0, 0.0

            # Meta-training
            with torch.backends.cudnn.flags(enabled=False):
                learner = maml.clone().double()
                db_train = DataLoader(fluxnet_train, batch_size=hyper_args["batch_size"], shuffle=True)
                max_steps = min(hyper_args["max_meta_step"], len(db_train))

                for step, (x, y) in enumerate(db_train):
                    x, y = x.to(device), y.to(device)
                    support_x, query_x, support_y, query_y = train_test_split(x, y, test_size=0.2, random_state=42)
                    pred = _get_pred(support_x, learner, with_context)
                    error = loss(pred, support_y)
                    learner.adapt(error)

                    # Evaluation
                    pred = _get_pred(query_x, learner, with_context)
                    error = loss(pred, query_y)
                    error.backward(retain_graph=True)
                    train_error += error.item()

                    if (step + 1) == max_steps:
                        break
                
                _grad_step(with_context)

                # Meta-testing
                learner = maml.clone(first_order=True).double()
                db_test = DataLoader(fluxnet_test, batch_size=hyper_args["batch_size"], shuffle=True)
                max_steps = min(hyper_args["max_meta_step"], len(db_test))
                
                for step, (x, y) in enumerate(db_test):
                    x, y = x.to(device), y.to(device)
                    support_x, query_x, support_y, query_y = train_test_split(x, y, test_size=hyper_args["finetune_size"], random_state=42)
                    
                    # Finetuning
                    pred = _get_pred(support_x, learner, with_context)
                    error = loss(pred, support_y)
                    learner.adapt(error)

                    # Evaluation
                    pred = _get_pred(query_x, learner, with_context)
                    error = loss(pred, query_y)
                    val_error += error.item()

                    if (step + 1) == max_steps:
                        if (for_inference) and (epoch % int(hyper_args["epoch"] // 2)) == 0:
                            # Take gradient steps for the class finetuned
                            error.backward(retain_graph=True)
                            opt.step()
                        
                        train_epoch.append(train_error/(step + 1))
                        val_epoch.append(val_error/(step + 1))
                        break
            
            if verbose and ((epoch % 50 == 0) or (epoch == (hyper_args["epoch"] - 1))):
                print(f'Epoch: {epoch}, training loss: {train_epoch[epoch]}, validation loss: {val_epoch[epoch]}')

        meta_loss_metric.update({
            f"train_loss_{run}": np.sqrt(np.array(train_epoch)),
            f"val_epoch_{run}": np.sqrt(np.array(val_epoch))
        })

    return meta_loss_metric, maml, encoder

def train_base(runs, hyper_args, fluxnet_train, fluxnet_test, verbose=False):
    base_loss_metric = dict()

    for run in range(0,runs):
        model = metaflux.learner.Learner(input_size=hyper_args["input_size"], hidden_size=hyper_args["hidden_size"], model_type=model_type).double().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=hyper_args["update_lr"])
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hyper_args["epoch"])
        train_epoch, val_epoch = list(), list()

        for epoch in range(hyper_args["epoch"]):
            train_error, val_error = 0.0, 0.0

            # Training
            db_train = DataLoader(fluxnet_train, batch_size=hyper_args["batch_size"], shuffle=True)
            max_steps = min(hyper_args["max_meta_step"], len(db_train))

            for step, (x, y) in enumerate(db_train):
                opt.zero_grad()
                x, y = x.to(device), y.to(device)
                train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
                pred = model(train_x[:,:,:hyper_args["input_size"]])
                error = loss(pred, train_y)
                error.backward()
                opt.step()
                train_error += loss(model(test_x[:,:,:hyper_args["input_size"]]), test_y).item()

                if (step + 1) == max_steps:
                    schedule.step()
                    break
            

            # Testing
            db_test = DataLoader(fluxnet_test, batch_size=hyper_args["batch_size"], shuffle=True)
            max_steps = min(hyper_args["max_meta_step"], len(db_test))
            model_copy = copy.deepcopy(model)
            opt_copy = torch.optim.Adam(model_copy.parameters(), lr=hyper_args["update_lr"])

            for step, (x, y) in enumerate(db_test):
                opt_copy.zero_grad()
                x, y = x.to(device), y.to(device)
                train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=hyper_args["finetune_size"], random_state=42)
                pred = model_copy(train_x[:,:,:hyper_args["input_size"]])
                error = loss(pred, train_y)

                # Akin to finetuning/adaptation step in meta-learning
                error.backward()
                opt_copy.step()
                train_error += error.item()

                with torch.no_grad():
                    pred = model_copy(test_x[:,:,:hyper_args["input_size"]])
                    error = loss(pred, test_y)
                    val_error += error.item()
                
                if (step + 1) == max_steps:
                    train_epoch.append(train_error/(step + 1))
                    val_epoch.append(val_error/(step + 1))
                    del model_copy, opt_copy
                    break

            if verbose and ((epoch % 50 == 0) or (epoch == (hyper_args["epoch"] - 1))):
                print(f'Epoch: {epoch}, training loss: {train_epoch[epoch]}, validation loss: {val_epoch[epoch]}')

        base_loss_metric.update({
            f"train_loss_{run}": np.sqrt(np.array(train_epoch)),
            f"val_epoch_{run}": np.sqrt(np.array(val_epoch))
        })
        
    return base_loss_metric, model