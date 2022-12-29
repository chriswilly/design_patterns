"""

"""
from __future__ import annotations
import IPython
import logging
import numpy as np

from dataclasses import (
    dataclass,
    field
    )


@dataclass
class FFT:
    """
    1D Periodic FFT Grid & Guaussian curve smoothinbg factor sigma
    see this explaining zero-padding:
        https://ccrma.stanford.edu/~jos/mdft/FFT_Zero_Padded_Sinusoid.html
    """
    # array numpy inputs are pandas dataframe column series
    data_series:np.ndarray
    time_series:np.ndarray
    name:str = None
    # filter coefficient
    # smoothing_factor:float = 1.0
    # inferred from array input size
    count:int                   = field(init=False, repr=True)
    length:float                = field(init=False, repr=True)
    # calulated from inferred vals
    frequency_domain:np.ndarray = field(init=False, repr=False)
    # mutated signal after class instance constructed
    spectral_coeffs:np.ndarray  = field(init=False, repr=False)
    spectral_norm:float         = field(init=False, repr=False)
    spectral_cdf:np.ndarray     = field(init=False, repr=False)
    filtered_coeffs:np.ndarray  = field(init=False, repr=False)
    filtered_signal:np.ndarray  = field(init=False, repr=False)


    def __post_init__(self)->None:
        self.set_grid()
        self.run()


    def set_grid(self)->None:
        # take b = floor of log length vector and calc 2**(b+1)
        self.count  = 2**(1+int(np.log2(self.data_series.size)))
        self.length = self.count*(self.time_series[1] - self.time_series[0])
        self.frequency_domain = (1/self.length)*np.linspace(-self.count/2, -1+self.count/2, self.count)
        # TODO --> np.fft.fftfreq(n)


    def bandpass_filter_freq(self)->np.ndarray:
        pass


    def bandpass_filter_indx(self)->np.ndarray:
        pass


    @staticmethod
    def filter_gaussian_1D(
        x:np.ndarray,
        x0:float=0.0,
        sigma:float=1.0
        )->np.ndarray:
        """
        1D gaussian filter with coordinate shift
        """
        return np.exp(-((x-x0)**2)/(2*sigma**2))


    def run(
        self,
        high_pass_filter:int = 0, # float(?) for freq spec
        low_pass_filter:int = -1
        )->None:
        """
        Fast Fourier Transform mutate self attrs
            spectral_coeffs
            filtered_coeffs
            filtered_signal
            spectral_cdf
        see this for module details:
            https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft
        """
        f_fft = np.fft.fft(
            self.data_series - np.mean(self.data_series), 
            n = self.count, 
            norm = 'ortho'
            )

        self.spectral_coeffs = np.fft.fftshift(f_fft) 
        self.spectral_norm   = np.linalg.norm(self.spectral_coeffs) 
        self.spectral_coeffs = self.spectral_coeffs / self.spectral_norm
        
        # mutate coefficients to apply band-pass-filter important mind your algebraic types: z = x+iy z* = x-iy
        temp_array = np.zeros(f_fft.shape, dtype = np.complex_)
        temp_array[high_pass_filter:low_pass_filter]   = f_fft[high_pass_filter:low_pass_filter]
        temp_array[-low_pass_filter:-high_pass_filter] = f_fft[-low_pass_filter:-high_pass_filter]
        
        # filtered signal for reconstruction low rank approx 
        self.filtered_coeffs = np.fft.fftshift(temp_array) 
        self.filtered_coeffs = self.filtered_coeffs / self.spectral_norm
        
        self.filtered_signal = np.sqrt(self.count)*np.fft.ifft(temp_array).real[:self.data_series.size] + np.mean(self.data_series)
        assert self.filtered_signal.size == self.time_series.size
        
        # create cumulative distriubution function for +Hz coefficients
        indx = self.frequency_domain >= 0
        
        # indx = np.ones(self.frequency_domain.size, dtype=bool)
        temp_array = np.zeros(indx.size)
        temp_array[indx] = np.abs(self.spectral_coeffs[indx])
        self.spectral_cdf = temp_array.cumsum()/temp_array.sum()



def test():
    """
    """
    frequency = (11,19,37)
    amplitude = (5e-2,4e-1,3e-3)

    array_length = 1_042 # row count
    time_limit = 10.1    # t_max s

    time_series   = time_limit/(array_length-1)*np.arange(0,array_length)
    signal = np.zeros(time_series.shape)
    for freq,amp in zip(frequency,amplitude):
            signal += amp*np.cos(time_series*freq*2*np.pi)

    test = FFT(
        data_series = signal,
        time_series = time_series
        )

    indx =(test.spectral_coeffs>=5*10**-3.5)

    IPython.embed()


if __name__=='__main__':
    test()
