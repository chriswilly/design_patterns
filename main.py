"""
"""
import sys
import pathlib
import argparse
import numpy as np
import IPython

# this package
import data_science as ds
from sql import cte_example as sql
from lib import utils


def fft_test()->bool:
    """
    """
    frequency = (5,22,47)

    amplitude = (5e-1,4e-1,3e-1)

    array_length = 1_042 # row count
    time_limit = 10.1    # t_max s

    time_series   = np.linspace(0,time_limit,array_length)

    signal = np.zeros(time_series.shape)

    for freq,amp in zip(frequency,amplitude):
        print(f'{amp}*cos(2*pi*{freq})')
        signal += amp*np.cos(time_series*freq*2*np.pi)

    test = ds.FFT(
        data_series = signal,
        time_series = time_series
        )

    indx = ds.peak_finder(curve=np.abs(test.spectral_coeffs),smoothing_factor=33)
    print(test.frequency_domain[indx])
    assert test.frequency_domain[indx].size//2 == 3

    return True


def graph_laplacian_test()->bool:
    """
    """
    raw = np.zeros([9,9])
    raw[:3,:3] = -1
    raw[-3:,-3:] = 1
    raw[-3:,:3] = 2
    resolution = 0.1
    # rng = np.random.default_rng()

    test = ds.GraphLaplacian(
        data=raw,
        distance_ratio=resolution
        )
    indx = (test.eigval<=5e-5)
    sum_zeros = np.sum(indx)
    print(f'{sum_zeros} distict groups given {resolution} relative distance')

    assert sum_zeros == 3
    return True


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

    test_GL = graph_laplacian_test()
    logger.info(f'{test_GL} graph_laplacian_test Passed')

    test_FFT = fft_test()
    logger.info(f'{test_FFT} fft_test Passed')

    # IPython.embed()


if __name__=='__main__':
    args = argparse.Namespace(**{'logs':'./logs','env':'dev','config':'./template_config.json'})
    main(args=args)
