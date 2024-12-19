import json
import os

def build_config(args, defaults=None):
    """
    Build configuration dictionary from defaults and CLI arguments.
    
    Args:
        args: ArgumentParser namespace containing CLI arguments
        defaults (dict, optional): Default configuration values
        
    Returns:
        tuple: (config_dict, preproc_kwargs)
    """
    if defaults is None:
        defaults = {
            'filename': '../arrhythmia_raw_data/MIT-BIH_raw.csv',
            'drop_classes': ['F'],
            'output_dir': os.getcwd(),
            'n_samples': -1,
            'binary': True,
            'scaler_name': 'MeanVariance'
        }

    # Load config file first
    config = defaults.copy()
    if args.config:
        with open(args.config, 'r') as file:
            config.update(json.load(file))

    # Override config with CLI arguments
    for key, value in vars(args).items():
        if (key != 'config' and 
            value is not None and 
            not (isinstance(value, bool) and value == False)):
            config[key] = value

    # Extract preprocessing parameters
    preproc_kwargs = {
        'drop_classes': config['drop_classes'],
        'n_samples': config['n_samples'],
        'binary': config['binary'],
        'scaler_name': config['scaler_name']
    }

    print(f"Using final config:\n{config}")
    
    return config, preproc_kwargs