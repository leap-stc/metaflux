def get_hyperparams():
    """
    Retrieve the set of hyperparameters used to fit and evaluate the model
    """
    args = {}
    args["xcolumns"] = ["P_F", "lai", "RH"]     # available list ["TA_F", "P_F", "lai", "RH", "WS_F", "PA_F", "NETRAD","RH","SWC"]
    args["epoch"] = 25      
    args["n_way"] = 2                          # define the number of flux stations to use
    args["k_spt"] = 250                         # the number of support data (similar to a training set)
    args["k_qry"] = 250                         # the number of query data (similar to a testing set)
    args["input_size"] = len(args["xcolumns"])  # the number of input features
    args["hidden_size"] = 64                    
    args["task_num"] = 1                        # similar to batchsize
    args["meta_lr"] = 1e-5      
    args["update_lr"] = 1e-5
    args["update_step"] = 250                   # the number of update step during meta-learning
    args["update_step_test"] = 5                # the number of adaptation steps (too high: very adapted, too low: not as adapted)
    args["num_lstm_layers"] = 2                 # the number of layers in the LSTM layers, one of [1,2]

    return args


def get_config(model, args=None):
    """
    Get model configuration for the baselearners

    Inputs:
    -------
    model: str
        The model type, one of ["mlp", "lstm", "bilstm"]
    args: dict
        The hyperparameters to specify the model, defaults to None

    Returns:
    config: list
        The model specification for use in the meta-learning routine
    """
    assert model in ["mlp", "lstm", "bilstm"], "model type is invalid... chose a valid one e.g. bilstm"

    if args is None:
        args = get_hyperparams()

    config = []

    if model == "lstm":
        config = [
            ("lstm", [args["hidden_size"]*4,args["input_size"],args["hidden_size"]]),
            ("leakyrelu", [0.01]),
            ("linear", [1,args["hidden_size"]])
        ]

    elif model == "mlp":
        config = [
            ("linear", [args["hidden_size"],args["input_size"]]),
            ("leakyrelu", [0.01]),
            ("linear", [args["hidden_size"],args["hidden_size"]]),
            ("leakyrelu", [0.01]),
            ("linear", [1,args["hidden_size"]])
        ]

    else:
        config = [
            ("bilstm", [args["hidden_size"]*4,args["input_size"],args["hidden_size"]]),
            ("leakyrelu", [0.01]),
            ("linear", [1,2*args["hidden_size"]])
        ]
    
    return config