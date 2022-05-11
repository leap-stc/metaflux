# Metaflux
Meta-learning framework for climate sciences.

## Quickstart
1. Clone this repository into your private workspace:
```
git clone https://github.com/juannat7/metaflux.git
```

2. Import the package into your notebook or IDE: 
```
import metaflux
import torch
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

3. Initialize the base-learner:
```
# load the hyperparameters specification and model base-learner model definition
hyper_args = metaflux.configs.get_hyperparams()
model_config = metaflux.configs.get_config(model="bilstm", args=hyper_args)

# initialize the base-learner
learner = metaflux.learner.Learner(config=model_config, input_size=hyper_args["input_size"], hidden_size=hyper_args["hidden_size"])

print(learner) # This should give you the structure of the base-model
```

4. Setting up DataLoader for batching and randomizing inputs at each iteration:
```
root_dir = <YOUR_DATA_DIRECTORY> # eg "./Data/tropics" which has the following folder structure ./Data/<class>/<mode>/<filenames>.csv
fluxnet_train = metaflux.dataloader.Fluxmetanet(root=root_dir, mode="train", batchsz=1, n_way=hyper_args["n_way"], k_shot=hyper_args["k_spt"], k_query=hyper_args["k_qry"], x_columns=hyper_args["xcolumns"], y_column='LE_CORR', time_column="TIMESTAMP_START", time_agg="1H", seasonality=7)

fluxnet_test = metaflux.dataloader.Fluxmetanet(root=root_dir, mode="test", batchsz=1, n_way=hyper_args["n_way"], k_shot=hyper_args["k_spt"], k_query=hyper_args["k_qry"], x_columns=hyper_args["xcolumns"], y_column='LE_CORR', time_column="TIMESTAMP_START", time_agg="1H", seasonality=7)
```

5. Training metalearner given our base-learner and dataloader:
```
maml = metaflux.metalearner.Meta(hyper_args, model_config)

# training
from torch.utils.data import DataLoader

db_train = DataLoader(fluxnet_train, hyper_args["task_num"], shuffle=True, num_workers=0)
for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_train):
    x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
    loss = maml(x_spt, y_spt, x_qry, y_qry)

# evaluation
db_test = DataLoader(fluxnet_test, 1, shuffle=True, num_workers=0)
for x_spt, y_spt, x_qry, y_qry in db_test:
    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
    loss = maml.finetuning(x_spt, y_spt, x_qry, y_qry)
```

## TO-DO:
- [x] Initializing base-learner and model specification
- [x] Data loader
- [x] Meta-learning routine
- [x] Training, validation loop
- [ ] Sample data