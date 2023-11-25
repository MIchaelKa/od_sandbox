import logging

def create_logger(name):
    logger = logging.getLogger(name)

    consoleHandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    consoleHandler.setFormatter(formatter)

    logger.propagate = False
    logger.handlers.clear()
    logger.addHandler(consoleHandler)

    return logger

logger = create_logger(__name__)


from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def create_tb_writer(experiment_name, dataset_name):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_folder = f'{current_time}_{experiment_name}'
    writer = SummaryWriter(f'./runs/{dataset_name}/{tensorboard_folder}/')
    return writer