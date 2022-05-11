import yaml

def get_hyperparams(config_path='metaflux/configs/hyperparams.yaml'):
    """
    Retrieve the set of hyperparameters used to fit and evaluate the model

    Parameters:
    -----------
    config_path <str>: the path to the hyperparameters YAML file

    Returns:
    --------
    args <dict>: dictionary containing the necessary values for hyperparameters
    """

    with open(config_path, 'r') as stream:
        args = yaml.safe_load(stream)

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