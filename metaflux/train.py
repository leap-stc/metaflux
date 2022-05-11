"""
Sample script to compar meta-learning. Feel free to build and define your own process

Sample usage: 
    python train.py -i "./data/tropics", -t "GPP_NT_VUT_REF" -m "bilstm"
"""
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from configs import *
from dataloader import *
from learner import *
from metalearner import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help = "Root directory of your inputs", required=True)
parser.add_argument("-t", "--target", default="GPP_NT_VUT_REF", help = "Target variable")
parser.add_argument("-m", "--model", default="mlp", help = "Models to use, currently only supports mlp, lstm, and bilstm")
cli_args = parser.parse_args()


hyper_args = get_hyperparams()
model_config = get_config(model=cli_args.model, args=hyper_args)

runs = 1

"""
Training baseline models
"""
print(f"Training meta-learning models")
vals_meta_losses = []
for run in range(0,runs):
    print(f"Run: {run + 1}")
    vals_meta_loss = []

    maml = Meta(hyper_args, model_config).to(device)
    print(f"Model specifications: {maml}")
    
    fluxnet_train = Fluxmetanet(root=cli_args.input, mode="train", batchsz=1, n_way=hyper_args["n_way"], k_shot=hyper_args["k_spt"], k_query=hyper_args["k_qry"], x_columns=hyper_args["xcolumns"], y_column=cli_args.target, time_column="TIMESTAMP_START", time_agg="1H", seasonality=7)
    fluxnet_test = Fluxmetanet(root=cli_args.input, mode="test", batchsz=1, n_way=hyper_args["n_way"], k_shot=hyper_args["k_spt"], k_query=hyper_args["k_qry"], x_columns=hyper_args["xcolumns"], y_column=cli_args.target, time_column="TIMESTAMP_START", time_agg="1H", seasonality=7)

    for epoch in range(hyper_args["epoch"]):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(fluxnet_train, hyper_args["task_num"], shuffle=True, num_workers=0)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            loss = maml(x_spt, y_spt, x_qry, y_qry)

            print('Epoch:', epoch, '\tTraining loss:', loss)

            if epoch % 2 == 0:  # evaluation
                db_test = DataLoader(fluxnet_test, 1, shuffle=True, num_workers=0)

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    loss = round(maml.finetuning(x_spt, y_spt, x_qry, y_qry), 6)
                    if epoch > 0 and loss > vals_meta_loss[-1]:
                        maml.update_lr = maml.update_lr * 0.1
                        
                    vals_meta_loss.append(loss)

                print('\tValidation loss:', loss)
                
    
    vals_meta_losses.append(vals_meta_loss)

"""
Training baseline models
"""
print("Training baseline models")
vals_base_losses = []
for run in range(0,runs):
    basemeta = BaseMeta(hyper_args["input_size"], hyper_args["hidden_size"], arch=cli_args.model).to(device)
    train_x, train_y = generate_base_metadata(root=cli_args.input, mode="test", batchsz=1, n_way=hyper_args["n_way"], k_shot=hyper_args["k_spt"], k_query=hyper_args["k_qry"], x_columns=hyper_args["xcolumns"], y_column=cli_args.target, time_column="TIMESTAMP_START", time_agg="1H", seasonality=7)
    test_x, test_y = generate_base_metadata(root=cli_args.input, mode="test", batchsz=1, n_way=hyper_args["n_way"], k_shot=hyper_args["k_spt"], k_query=hyper_args["k_qry"], x_columns=hyper_args["xcolumns"], y_column=cli_args.target, time_column="TIMESTAMP_START", time_agg="1H", seasonality=7)
    
    optimizer = torch.optim.Adam(basemeta.parameters(), lr=hyper_args["update_lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=0)
    criterion = torch.nn.MSELoss().to(device)
    vals_base_loss = []

    for epoch in range(hyper_args["epoch"]):
        basemeta = basemeta.train()
        train_losses = []

        for i, x in enumerate(train_x):
            pred = basemeta(x.unsqueeze(0).float().to(device))
            loss = criterion(pred.squeeze()[-1], train_y[i].float().to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.array(train_losses)
        train_loss = np.mean(abs(train_loss - train_loss.mean()) / train_loss.std())
        print('Epoch:', epoch, '\tTraining loss:', train_loss)

        if epoch % 2 == 0:
            basemeta = basemeta.eval()
            val_losses = []

            with torch.no_grad():
                for i, x in enumerate(test_x):
                    pred = basemeta(x.unsqueeze(0).float().to(device))
                    loss = criterion(pred.squeeze()[-1], test_y[i].float().to(device))
                    val_losses.append(loss.item())

            val_loss = np.array(val_losses)
            val_loss = np.mean(abs(val_loss - val_loss.mean()) / val_loss.std())
            scheduler.step(val_loss)
            vals_base_loss.append(val_loss)

            print('\tValidation loss:', val_loss)

    vals_base_losses.append(vals_base_loss)

# Plot validation loss
f, ax = plt.subplots()
epochs = np.arange(1,hyper_args["epoch"] + 1, 2)
ax.plot(epochs, vals_meta_losses[0], label='Meta-learning')
ax.plot(epochs, vals_base_losses[0], label='Non Meta-learning')
ax.set_title("GPP inference with and without meta-learning")
ax.set_ylabel("MSE")
ax.set_xlabel("Epoch")
ax.legend()
plt.show()