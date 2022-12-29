"""
"""
import numpy as np
from scipy.signal import find_peaks_cwt
from scipy.ndimage import gaussian_filter1d


def peak_finder(
    curve:np.ndarray,
    smoothing_factor:float=21.0,
    )->np.ndarray:
    """
    """
    min_width = int(curve.size/20)
    max_width = int(curve.size/5)
    resolution = int((max_width - min_width)/19)
    peak_width = np.arange(min_width,max_width,resolution)

    new_curve = gaussian_filter1d(curve,sigma=smoothing_factor)

    indx = find_peaks_cwt(new_curve,peak_width)

    return indx
