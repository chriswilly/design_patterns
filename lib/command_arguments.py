"""
"""
from __future__ import annotations
import argparse

import utils

def main(args:argparser.Namespace)->None:
    """
    """
    logger,_ = utils.start_log(name='test',destination=args.logs)
    logger.info(f'input params\n{vars(args)}')


if __name__=='__main__':
    """
    look at data root dir, 
    if not reachable call tkinker dialog input for path,
    enumerate root contents of file ext key and feed to main process
    """
    parser = argparse.ArgumentParser(description='Data Directory & Config Path:')

    parser.add_argument(
        '--env', metavar='runtime environment',
        type=str, nargs='?',
        help='runtime Enum environment dev, test, prod',
        default='dev'
        )

    parser.add_argument(
        '--config', metavar='config file',
        type=str, nargs='?',
        help='user config.json path',
        default='../config.json'
        )

    parser.add_argument(
        '--logs', metavar='log output directory',
        type=str, nargs='?',
        help='logs output path',
        default='../logs'
        )

    parser.add_argument(
        '--data', metavar='target folder',
        type=str, nargs='?',
        help='root path',
        default='../data'
        )

    args = parser.parse_args()
    main(args=args)
