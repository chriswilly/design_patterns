"""
"""
import sys
import pathlib
import argparse
import IPython
sys.path.append(pathlib.Path(__file__).resolve().parents[1].__str__())
print(sys.path)

# if __package__==None:
#     __package__='design_patterns'

from design_patterns.data_science import fft
from design_patterns.data_science import graph_laplacian as gl

from design_patterns.sql import cte_example
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

    IPython.embed()

if __name__=='__main__':
    args = argparse.Namespace(**{'logs':'../logs','env':'dev','config':'../template_config.json'})
    main(args=args)
