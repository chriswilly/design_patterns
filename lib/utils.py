"""
"""

import sys
import logging
import pathlib
import datetime
import json
from enum import Enum

class Environment(Enum):
    """
    Enumerated keys for runtime specification into config keys
    """
    development = 0
    dev = 0
    test = 1
    production = 2
    prod = 2


def time_string()->str:
    return datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')


def get_config(configpath:pathlib.Path,env:Environment)->dict:
    """
    using Environment object to map a finite set to dbconfig keys
    e.g. Environment.dev.name = development as {0:{development,dev,...},1:...}
    """
    with open(configpath) as file:
        raw = json.load(file)

    config = {}
    for entry in raw.keys():
        try:
            config[Environment[entry].name] = raw.get(entry)
        except:
            print(f'Outer json key {entry} is not in Environment(Enum) expected vales, skipped.')

    return config[env.name]


def start_log(
    name:str,
    destination:str, # ->pathlib.Path (?)
    caller:str = __name__,
    ext:str = '.log'
    )->tuple[logging.Logger,pathlib.Path]:
    """ create log directory, return logger obj and path
    """
    timestamp = time_string()
    
    log_name = f'{destination}/{name}_{timestamp}'

    log_file = pathlib.Path(log_name).with_suffix(ext).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO
        )
    
    logger = logging.getLogger(caller)
    handler = logging.StreamHandler(sys.stdout)
    # handler = logging.FileHandler(log_file)  
    logging_formatter = logging.Formatter(r'%(asctime)s:%(name)s:%(message)s')
    handler.setFormatter(logging_formatter)
    logger.addHandler(handler)

    return logger, log_file


if __name__=='__main__':
    print('call externally')
    raise(NotImplementedError)
