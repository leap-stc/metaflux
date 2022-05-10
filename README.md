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

fluxnet_test = metaflux.dataloader.Fluxmetanet(root=root_dir, mode="test", batchsz=1, n_way=hyper_args["n_way"], k_shot=hyper_args["k_spt"], k_query=hyper_args["k_qry"], x_columns=hyper_args["xcolumns"], y_column='LE_CORR', time_column="TIMESTAMP_START", time_agg="1H", seasonality=7)
```

## TO-DO:
- [x] Initializing base-learner and model specification
- [x] Data loader
- [ ] Meta-learning routine
- [ ] Training, validation loop
- [ ] Sample data