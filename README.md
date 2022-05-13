# Metaflux
Meta-learning framework for climate sciences. Currently supports the following features:
- Takes as input timeseries data (eg. FLUXNET eddy covariance stations)
- Customizable MLP, LSTM, and BiLSTM models (update the `configs.py` file) TODO: customizable with YAML file
- Customizable hyperparameters (create your own and place them under the `configs` directory)
- Sample training script with sample data that can be adapted to your own use case

## Quickstart
1. Clone this repository into your private workspace:
```
git clone https://github.com/juannat7/metaflux.git
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. On terminal, you can run a sample training script using FLUXNET stations data, meta-learned on the tropics
```
cd metaflux
python train.py -i "./data/tropics", -t "GPP_NT_VUT_REF"
```

![Meta inference](https://github.com/juannat7/metaflux/blob/main/docs/gpp_infer.jpeg)

## Build-Your-Own Metalearning Pipeline
You can customize the package by changing the hyperparameters and base-learners
1. Import the package into your notebook or IDE: 
```
from metaflux import metaflux
import torch
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

2. Load hyperparametersthe base-learner:
```
hyper_args = metaflux.configs.get_hyperparams()
```

3. Initialize base-learner
```
model_config = metaflux.configs.get_config(model="bilstm", args=hyper_args)
learner = metaflux.learner.Learner(config=model_config, input_size=hyper_args["input_size"], hidden_size=hyper_args["hidden_size"])
```

4. Setting up DataLoader for batching and randomizing inputs at each iteration:
```
root_dir = <YOUR_DATA_DIRECTORY> # eg "metaflux/data/tropics" which has the following folder structure ./Data/<class>/<mode>/<filenames>.csv
fluxnet_train = metaflux.dataloader.Fluxmetanet(root=root_dir, mode="train", batchsz=1, n_way=hyper_args["n_way"], k_shot=hyper_args["k_spt"], k_query=hyper_args["k_qry"], x_columns=hyper_args["xcolumns"], y_column='GPP_NT_VUT_REF', time_column="TIMESTAMP_START", time_agg="1H", seasonality=7)

fluxnet_test = metaflux.dataloader.Fluxmetanet(root=root_dir, mode="test", batchsz=1, n_way=hyper_args["n_way"], k_shot=hyper_args["k_spt"], k_query=hyper_args["k_qry"], x_columns=hyper_args["xcolumns"], y_column='GPP_NT_VUT_REF', time_column="TIMESTAMP_START", time_agg="1H", seasonality=7)
```

5. Training metalearner given our base-learner and dataloader:
```
maml = metaflux.metalearner.Meta(hyper_args, model_config)

# training
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
- [x] Sample data and baseline evaluation
- [x] Abstract hyperparameters and model configurations as modifiable YAML
- [ ] Allow customizable baseline models
