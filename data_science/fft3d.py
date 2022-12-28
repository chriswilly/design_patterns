"""

"""
from __future__ import annotations
import IPython
import logging
import numpy as np
import pandas as pd
from dataclasses import (
    dataclass,
    field
    )



@dataclass
class FFT3D:
    """
    3D organized n-dimensional structure, used in 1D vectors and 3D mesh
    """
    # scalar
    length:int = 10
    count:int = 64
    smoothing: float = 2

    # -10 to +10 with 64 elements
    x: np.ndarray = np.linspace(-length,length,count+1)[0:count]
    y: np.ndarray = np.linspace(-length,length,count+1)[0:count]
    z: np.ndarray = np.linspace(-length,length,count+1)[0:count]

    # 3x64x64x64 array
    grid_mesh: np.ndarray   = np.array(np.meshgrid(x, y, z))
    # -pi/L to pi/L centered about 0 --> plot with fftshift (fourier) & ifftshift (real)
    frequency: np.ndarray   = (np.pi/length)*np.linspace(-count/2, -1+count/2, count)
    # 3x64x64x64 array
    frequencies: np.ndarray = np.array(np.meshgrid(frequency,frequency,frequency))


def locate_max(array:np.ndarray)->tuple:
    """
    return max index of assumed 3D signal
    This method mixes x-y coordinates up
    """
    if not len(array.shape)<=3 & len(array.shape)>0:
        raise ValueError(f'signal dimension {array.shape} not supported')

    else:
        return np.unravel_index(array.argmax(),array.shape)  #order='F')



def find_peak_frequency(
    data:np.ndarray,
    signal_init:np.ndarray,   # (_time_,_xdata_,_ydata_,_zdata_)
    )->tuple:
    """
    Shift to Fourier space and take average over all timepoints to identifiy
    signal target frequency indicies
    return two index pairs corresponing to +k and -k values
    """

    # take fft of each slice in time
    for timepoint in np.arange(data.shape[1]):
        f_input = data[:,timepoint]                     # Vector
        f_cube = f_input.reshape(signal_init.shape[1:]) # 3D array

        # shifted so boundary frequencies at center
        fft_data = np.fft.fftshift(np.fft.fftn(f_cube))

        signal_init[timepoint,...] = fft_data

    # keep sign in normalization
    signal_norm = signal_init/np.abs(signal_init).max()

    # signal_mean = np.fft.ifftn(
    #                 np.fft.ifftshift(
    #                 signal_norm.mean(axis=0))
    #                 )

    # average in freq domain across all timesteps
    signal_mean = signal_norm.mean(axis=0)
    signals = (signal_mean, signal_norm)


    peak_coordinates = locate_max(signal_mean)

    return peak_coordinates, signals


def filter_gaussian(
    x:np.ndarray,
    y:np.ndarray,
    z:np.ndarray,
    sigma:float,
    x0:float=0,
    y0:float=0,
    z0:float=0
    )->np.ndarray:
    """
    3D gaussian filter with coordinate shift, default origin (0,0,0)
    intended to be applied in fourier freq space kx, ky, kz mesh grids
    """
    return np.exp(-((x-x0)**2 + (y-y0)**2 + (z-z0)**2)/(2*sigma**2))



def fourier_transform_denoise(
    f_input:np.ndarray,
    filter_values:np.ndarray=None
    )->np.ndarray:
    """
    apply fftn to 3D array input as vector

    """
    # class example
    # noisy_face_hat = np.fft.fft2(noisy_face)
    # noisy_face_hat = np.fft.fftshift(noisy_face_hat)
    # noisy_face_hat_filtered = g_vals*noisy_face_hat
    # noisy_face_filtered = np.real(
    #                       np.fft.ifft2(
    #                       np.fft.ifftshift(
    #                           noisy_face_hat_filtered) ))

    f_cube = f_input.reshape(filter_values.shape)
    f_fft = np.fft.fftn(f_cube)
    f_hat = np.fft.fftshift(f_fft)

    f_hat_filtered = f_hat*filter_values

    f_fft_filtered = np.fft.ifftshift(f_hat_filtered)

    return np.fft.ifftn(f_fft_filtered).real




def run(
    data:np.ndarray,
    grid:FFT3D,
    IMAGE_EXT:str = 'png'
    )->bool:
    """
    """

    # shape:  (_time_,_xdata_,_ydata_,_zdata_)
    signal = np.zeros([data.shape[1],*grid.frequencies.shape[1:]])
    signal_complex = np.zeros([data.shape[1],*grid.frequencies.shape[1:]],dtype=np.complex_)

    trajectory = np.zeros([data.shape[1],grid.frequencies.shape[0]])

    # todo: make symmetric with -k feqs for filter gaussian placement
    indx_peak, (signal_mean, _) = find_peak_frequency(
        data=data,
        signal_init=signal_complex
        )

    # switched x,y plot position:
    # % np.unravel_index(gaussian_positive_vals.argmax(),(64,64,64))
    # % (47, 38, 9)  ...?

    print(signal_mean[...,indx_peak[2]].shape)


    print('peak signal index in Fourier space:\n',indx_peak)
    print(np.array([
          grid.frequency[indx_peak[0]],
          grid.frequency[indx_peak[1]],
          grid.frequency[indx_peak[2]]
          ])
          )

    gaussian_positive_vals = filter_gaussian(
        x = grid.frequencies[0,...],
        y = grid.frequencies[1,...],
        z = grid.frequencies[2,...],
        sigma = grid.smoothing,
        x0=grid.frequency[indx_peak[1]],
        y0=grid.frequency[indx_peak[0]],
        z0=grid.frequency[indx_peak[2]]
        )
    # to maintain imaginary part upon reconstruction
    gaussian_negative_vals = filter_gaussian(
         x = grid.frequencies[0,...],
         y = grid.frequencies[1,...],
         z = grid.frequencies[2,...],
         sigma = grid.smoothing,
         x0=grid.frequency[-indx_peak[1]],
         y0=grid.frequency[-indx_peak[0]],
         z0=grid.frequency[-indx_peak[2]]
        )
    # compose array of gaussians around 2 real/imaginary points in kx,ky,kz
    gaussian_filter_vals = gaussian_positive_vals + gaussian_negative_vals

    # this will be at max z

    for timepoint in np.arange(data.shape[1]):
        smoothed_data = fourier_transform_denoise(
            f_input = data[:,timepoint],
            filter_values = gaussian_filter_vals
            )
            #.reshape(grid.frequencies.shape[1:])
        signal[timepoint,...] = smoothed_data
        coordinates = locate_max(smoothed_data)
        trajectory[timepoint,...] = (
            grid.frequency[coordinates[1]],
            grid.frequency[coordinates[0]],
            grid.frequency[coordinates[2]]
            )
        print(f'loc at t{str(timepoint).zfill(2)}: {locate_max(smoothed_data)}')

    signal_norm = np.abs(signal)/np.abs(signal).max()

    return signal, signal_norm



def main():
    """
    """
    data = load_data('*.npy')
    grid = FFT3D()
    signal, signal_norm = run(data,grid)


if __name__=='__main__':
    main()
