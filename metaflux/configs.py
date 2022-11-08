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

    if args["input_size"] != len(args["xcolumns"]):
        args["input_size"] = len(args["xcolumns"])

    return args