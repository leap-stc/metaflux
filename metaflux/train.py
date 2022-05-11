"""
Sample script to train meta-learning. Feel free to build and define your own process

Sample usage: 
    python train.py -i "./Data/", -t "LE_CORR"
"""
from torch.utils.data import DataLoader

from configs import *
from dataloader import *
from learner import *
from metalearner import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help = "Root directory of your inputs", required=True)
parser.add_argument("-t", "--target", default="LE_CORR", help = "Target variable")
cli_args = parser.parse_args()


hyper_args = get_hyperparams()
model_config = get_config(model="bilstm", args=hyper_args)
fluxnet_test = Fluxmetanet(root=cli_args.input, mode="test", batchsz=1, n_way=hyper_args["n_way"], k_shot=hyper_args["k_spt"], k_query=hyper_args["k_qry"], x_columns=hyper_args["xcolumns"], y_column=cli_args.target, time_column="TIMESTAMP_START", time_agg="1H", seasonality=7)

val_meta_losses = []
for run in range(0,10):
    print(f"Run: {run + 1}")
    val_loss = []

    maml = Meta(hyper_args, model_config).to(device)
    print(f"Model specifications: {maml}")
    
    fluxnet = Fluxmetanet(root=cli_args.input, mode="train", batchsz=1, n_way=hyper_args["n_way"], k_shot=hyper_args["k_spt"], k_query=hyper_args["k_qry"], x_columns=hyper_args["xcolumns"], y_column=cli_args.target, time_column="TIMESTAMP_START", time_agg="1H", seasonality=7)

    for epoch in range(hyper_args["epoch"]):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(fluxnet, hyper_args["task_num"], shuffle=True, num_workers=0)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            loss = maml(x_spt, y_spt, x_qry, y_qry)

            print('Epoch:', epoch, '\ttraining loss:', loss)

            if epoch % 2 == 0:  # evaluation
                db_test = DataLoader(fluxnet_test, 1, shuffle=True, num_workers=0)

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    loss = round(maml.finetuning(x_spt, y_spt, x_qry, y_qry), 6)
                    if epoch > 0 and loss > val_loss[-1]:
                        print(f"Updating lr...")
                        maml.update_lr = maml.update_lr * 0.1
                        
                    val_loss.append(loss)

                print('test loss:', loss)
                
    
    val_meta_losses.append(val_loss)