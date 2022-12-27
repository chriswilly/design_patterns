"""
"""
import sys
import pathlib
import argparse
import IPython
sys.path.append(pathlib.Path(__file__).resolve().parents[2].__str__())
# print(sys.path)
# IPython.embed()

# this package
import design_patterns.data_science as ds
from design_patterns.sql import cte_example as sql
from design_patterns.lib import utils


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
    print(sys.path[0])
    print(sys.path[-1])

    IPython.embed()

if __name__=='__main__':
    args = argparse.Namespace(**{'logs':'../logs','env':'dev','config':'../template_config.json'})
    main(args=args)
