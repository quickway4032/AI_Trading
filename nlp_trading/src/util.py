#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import yaml
import logging
logger = logging.getLogger('nlp')

def get_configs():
    
    with open('./src/default.yml') as f:
        return yaml.unsafe_load(f)
        
def get_logger(file_name, logging_level,logs_directory):
   
    logs_directory=Path(logs_directory)
    logs_directory.mkdir(exist_ok=True,parents=True)
    logger.setLevel(logging_level)
    fh = logging.FileHandler(logs_directory/file_name, mode='w')
    fh.setLevel(logging_level)
    sh = logging.StreamHandler()
    sh.setLevel(logging_level)
    logging_formatter = logging.Formatter("%(asctime)s:[%(levelname)s]:[%(name)s]:%(message)s")
    fh.setFormatter(logging_formatter)
    sh.setFormatter(logging_formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger