#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import model_estimation
import logging
import argparse
from util import *

logger = logging.getLogger('nlp')

class ArgumentDefaultsHelpFormatter_RawDescription(argparse.ArgumentDefaultsHelpFormatter):
    def _fill_text(self, text, width, indent):
        return ''.join(indent + line for line in text.splitlines(keepends=True))

def parse(args=None):
    parser = argparse.ArgumentParser(description='Conduct Sentiment Analysis')
    parser.add_argument('--data_directory', default=Path('data'), type=Path, nargs='?', help='data directory (default ./data).')
    parser.add_argument('--output_directory', default=Path('output'), type=Path, nargs='?', help='Output directory (default ./output).')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Verbose (add output);  can be specificed multiple times to increase verbosity')
        
    return parser.parse_args(args) 

def run(args):
    log_directory = args.output_directory/'logs'
    log_directory.mkdir(exist_ok=True, parents=True)
    model_estimation.run(args.data_directory,args.output_directory)
    logger.debug('The processing is sucessfully completed.')
        
def main():   
    args = parse()

    if 0 == args.verbose:
        ll = logging.ERROR
    elif 1 == args.verbose:
        ll = logging.INFO
    else:
        ll = logging.DEBUG

    logs_directory = args.output_directory/'logs'
    logs_directory.mkdir(exist_ok=True,parents=True)
    
    get_logger('nlp.log',ll, logs_directory)
    
    logger.setLevel(ll)
    
    fh = logging.FileHandler(logs_directory/f'nlp_trading.log', mode='w')
    sh = logging.StreamHandler()
    
    fh.setLevel(ll)
    sh.setLevel(ll)
    logging_formatter = logging.Formatter("%(asctime)s:[%(levelname)s]:[%(name)s]:%(message)s")
    fh.setFormatter(logging_formatter)
    sh.setFormatter(logging_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    run(args)                
 
if __name__ == '__main__':
    import sys
    sys.exit(main())
