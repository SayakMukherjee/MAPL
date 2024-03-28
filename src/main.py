#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 30-Jan-2023
# 
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

from utils import Config

import wandb
import json
import logging
import torch
import click
import random
import numpy as np

from pathlib import Path
from optim import get_method

@click.command()
@click.option('--exp_config', type=click.Path(exists=True), default=None,
              help='Configuration file for the experiment (default: None).')
def main(exp_config):

    # Load configuration file
    config = Config(locals().copy())
    config.load_config(Path(exp_config))

    config.settings['experiment_name'] = config.settings['experiment_name']

    # Update log path to current experiment folder
    config.settings['configurations']['log_path'] = Path(config.settings['configurations']['log_path']).joinpath(config.settings['type'])
    config.settings['configurations']['log_path'] = Path(config.settings['configurations']['log_path']).joinpath(config.settings['experiment_name'])

    # Set up logger
    if not Path.exists(Path(config.settings['configurations']['log_path'])):
        Path.mkdir(Path(config.settings['configurations']['log_path']), parents=True, exist_ok=True)

    if config.settings['train']:
        log_path = Path(config.settings['configurations']['log_path']).joinpath('log.txt')
    else:
        log_path = Path(config.settings['configurations']['log_path']).joinpath('log_test.txt')

    logging.basicConfig(level = logging.INFO,
                        filemode = 'w',
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename = log_path)
    logger = logging.getLogger()

    # logger.info('Log file is %s.' % (log_path))
    
    # Set seed
    if config.settings['configurations']['seed'] != -1:

        # if -1 then keep randomised
        random.seed(config.settings['configurations']['seed'])
        np.random.seed(config.settings['configurations']['seed'])
        torch.manual_seed(config.settings['configurations']['seed'])
        torch.cuda.manual_seed(config.settings['configurations']['seed'])
        torch.cuda.manual_seed_all(config.settings['configurations']['seed'])
        logger.info('Set seed to %d.' % config.settings['configurations']['seed'])

    # Default device to 'cpu' if cuda is not available
    if config.settings['configurations']['device'] == "cuda" and not torch.cuda.is_available():
        config.settings['configurations']['device'] = "cpu"

    logger.info(json.dumps(config.formatted_config(), indent=4))

    wandb_path = Path('../wandb')
    if not Path.exists(wandb_path):
        Path.mkdir(wandb_path, exist_ok=True, parents=True)

    # start a new wandb run to track this script
    wandb.init(
        mode="disabled", # "online", "offline", "disabled"
        # set the wandb project where this run will be logged
        project="mapl_" + config.settings["type"],
        name=config.settings['experiment_name'],
        entity="myentity",
        dir=wandb_path,
        # track hyperparameters and run metadata
        config=config.formatted_config()
    )

    get_method(config.settings['method'])(config)

    wandb.finish()

if __name__ == '__main__':
    main()