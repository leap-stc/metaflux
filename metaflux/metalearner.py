from copy import deepcopy
import numpy as np
from torch import nn
import warnings
warnings.filterwarnings("ignore")
try:
    from learner import *
except:
    from .learner import *

class Meta(nn.Module):
    """
    The main class for meta-learning, using base-learner and dataloader

    Parameters:
    -----------
    args <dict>: hyperparameter dictionary containing important parameters such as learning rates
    config <list>: list of tuples defining the structure of the base-learners
    """

    def __init__(self, args, config):
        super(Meta, self).__init__()
        
        self.update_lr = args["update_lr"]
        self.meta_lr = args["meta_lr"]
        self.n_way = args["n_way"]
        self.k_spt = args["k_spt"]
        self.k_qry = args["k_qry"]
        self.task_num = args["task_num"]
        self.update_step = args["update_step"]
        self.update_step_test = args["update_step_test"]
        
        self.net = Learner(config, input_size=args["input_size"], hidden_size=args["hidden_size"])
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optim, "min", patience=0)
        
    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        Forward routine for fitting at the meta-training phase

        Parameters:
        -----------
        x_spt: support set of input features with shape (batchsz, cls_num, k_shot, seq_len, n_features])
        y_spt: support set of target feature with shape (batchsz, cls_num)
        x_qry: query set of input features with shape (batchsz, cls_num, k_qry, seq_len, n_features)
        y_qry: query set of target feature with shape (batchsz, cls_num)
        """
        task_num, setsz, t_, n, f = x_spt.size()
        querysz = x_qry.size(1)
        k = 0
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

        for i in range(setsz):
            # 1. run the i-th task and compute loss for k=0
            preds = self.net(x_spt[:,i,k,:,:], vars=None)
            loss = F.mse_loss(preds.squeeze()[-1:], y_spt[:,i,0].squeeze())
            grad = torch.autograd.grad(loss, self.net.parameters(), allow_unused=True)
            fast_params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            fast_weights = nn.ParameterList()
            for layer in fast_params:
                fast_weights.append(nn.Parameter(layer))
            
            # this is the loss and accuracy before first update
            with torch.no_grad():
                preds_q = self.net(x_qry[:,i,k,:,:], self.net.parameters())
                loss_q = F.mse_loss(preds_q.squeeze()[-1:], y_qry[:,i,0])
                losses_q[0] += loss_q.item()

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q = self.net(x_qry[:,i,0,:,:], fast_weights)
                loss_q = F.mse_loss(logits_q.squeeze()[-1:], y_qry[:,i,0])
                losses_q[1] += loss_q.item()

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1 (ie. update_step)
                preds = self.net(x_spt[:,i,k,:,:], fast_weights)
                loss = F.mse_loss(preds.squeeze()[-1:], y_spt[:,i,k])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                fast_weights = nn.ParameterList()
                for layer in fast_params:
                    fast_weights.append(nn.Parameter(layer))

                preds_q = self.net(x_qry[:,i,k,:,:], fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.mse_loss(preds_q.squeeze()[-1:], y_qry[:,i,k])
                losses_q[k + 1] += loss_q.item()
                if k + 1 == self.update_step:
                    loss_q = loss_q / setsz

        # end of all tasks
        # sum over all losses on query set across all tasks
        losses_q = np.array(losses_q)
        losses_q = abs(losses_q - losses_q.mean()) / losses_q.std()

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()

        return losses_q[-1]


    def finetuning(self, x_spt, y_spt, x_qry, y_qry):
        """
        Finetuning step at the meta-testing phase

        x_spt: support set of input features with shape (cls_num, k_shot, seq_len, n_features])
        y_spt: support set of target feature with shape (cls_num)
        x_qry: query set of input features with shape (cls_num, k_qry, seq_len, n_features)
        y_qry: query set of target feature with shape (cls_num)
        """
        setsz, t_, n, f = x_spt.size()
        querysz = x_qry.size(0)
        losses_q = [0 for _ in range(self.update_step_test + 1)]
        k = 0

        assert len(x_spt.shape) == 4

        # in order to preserve the state of running_mean/variance, we are finetuning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        for i in range(setsz):
            preds_q = net(x_spt[i:i+1,k,:,:])
            loss = F.mse_loss(preds_q.squeeze()[-1:], y_spt[i,0].squeeze())
            grad = torch.autograd.grad(loss, net.parameters())
            fast_params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
            fast_weights = nn.ParameterList()
            for layer in fast_params:
                fast_weights.append(nn.Parameter(layer))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                preds_q = net(x_qry[i:i+1,k,:,:], net.parameters())
                loss_q = F.mse_loss(preds_q.squeeze()[-1:], y_qry[i,0])
                losses_q[0] += loss_q.item()

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                y_preds = net(x_qry[i:i+1,k,:,:], fast_weights)
                loss_q = F.mse_loss(y_preds.squeeze()[-1:], y_qry[i,0])
                losses_q[1] += loss_q.item()

            for k in range(1, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                preds = net(x_spt[i:i+1,k,:,:], fast_weights)
                loss = F.mse_loss(preds.squeeze()[-1:], y_spt[i,k])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                fast_weights = nn.ParameterList()
                for layer in fast_params:
                    fast_weights.append(nn.Parameter(layer))

                preds_q = net(x_qry[i:i+1,k,:,:], fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.mse_loss(preds_q.squeeze()[-1:], y_qry[i,k])
                losses_q[k + 1] += loss_q.item()

        del net
        # take the minimum of the last 3% of update_step size for numerical stability and convergence
        losses_q = np.array(losses_q)
        losses_q = abs(losses_q - losses_q.mean()) / losses_q.std()
        loss_q = losses_q[-1] / setsz

        return loss_q