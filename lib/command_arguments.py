"""
"""
from __future__ import annotations
import argparse

import utils

def main(args:argparse.Namespace)->None:
    """
    """
    logger,_ = utils.start_log(name='test',destination=args.logs)
    logger.info(f'input params\n{vars(args)}')
    config = utils.get_config(
        configpath=args.config,
        env=utils.Environment[args.env]
        )
    logger.info(f'{args.env} maps to config params\n{config}')


if __name__=='__main__':
    """
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
