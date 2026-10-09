
import logging 

def setup_logger(name:str,log_file:str,level=logging.INFO) -> logging.Logger:
    """
    Create and configure a logger

    Args:
        name(str):File to log to
        log_file(str):File to log to
        level:logging level
    
    """
    handler=logging.FileHandler(log_file)
    formatter=logging.Formatter('%(ascctime)s-%(name)s-%(levelname)s-%(message)s')
    handler.setFormatter(formatter)

    logger=logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

