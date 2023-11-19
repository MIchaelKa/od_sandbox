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