# Metaflux
Meta-learning framework for climate sciences.

## Quickstart
1. Clone this repository into your private workspace:
```
git clone https://github.com/juannat7/metaflux.git
```

2. Import the package into your notebook or IDE: 
```
from metaflux import *
```

3. Initialize the base-learner:
```
# load the hyperparameters specification and model base-learner model definition
hyper_args = metaflux.configs.get_hyperparams()
model_config = metaflux.configs.get_config(model="bilstm", args=hyper_args)

# load the base-learner
learner = metaflux.learner.Learner(config=model_config, 
                                    input_size=hyper_args["input_size"],
                                    hidden_size=hyper_args["hidden_size"]
                                    )

print(learner) # This should give you the structure of the base-model
```

## TO-DO:
- [x] Initializing base-learner and model specification
- [ ] Data loader
- [ ] Meta-learning routine
- [ ] Training, validation loop