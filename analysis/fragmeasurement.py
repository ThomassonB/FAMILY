from . import network_utility as utility
from . import analyse

import numpy as np
from scipy import stats

def estimated_f(f_up, f_low):
    return (f_up + f_low)/2, (f_up - f_low)/2

def phi2D(fractality):
    return np.log(fractality) / np.log(2)

def delta_phi2D(fractality, delta_fractality):
    return delta_fractality / (fractality * np.log(2))

def compute_fractality(network, version="default", mode=("HIERARCHICAL",), confidence_level=0.682689492):
    """
    version = default takes the median, compute the uncertainty on the median and do the same with holes corrected network. 
    Then final uncertainty is bounded with the lowest value with holes and the highest value without holes
    version = legacy is the same but does not take the median uncerainty
    mode is a tuple containing the fragmentation regimes you want to account for, (e.g. HIERARCHICAL, LINEAR)
    """
    
    virtual = utility.virtualNodes(network.network, network.levels)
    network_filled = analyse.Network(network.data, min_overlap=network.min_overlap, graph=virtual)
    utility.fractality(network_filled.network)
    
    network_filled.extractStructures()
        
    fractalities = [struct.fractality for struct in network.structures if struct.mode.name in mode]
    fractalities_filled = [struct.fractality for struct in network_filled.structures if struct.mode.name in mode]

    if version == "legacy":
        f_up = np.median(fractalities_filled)
        f_low = np.median(fractalities)
        
    elif version == "default":
        ic = stats.quantile_test(fractalities, p=0.5).confidence_interval(
            confidence_level=confidence_level
        )
        
        ic_filled = stats.quantile_test(fractalities_filled, p=0.5).confidence_interval(
            confidence_level=confidence_level
        )
    
        f_up = ic_filled.high
        f_low = ic.low
    
    return estimated_f(f_up, f_low)

def compute_phi2d(network, **kwargs):
    fractality, delta_fractality = compute_fractality(network, **kwargs)
    return phi2D(fractality), delta_phi2D(fractality, delta_fractality)

def phi3D(phi2d, relative_error):
    return phi2d / (1 + relative_error)

def delta_phi3D(phi2d, delta_phi2d, relative_error, delta_relative_error):
    return np.sqrt( (delta_phi2d / (1+relative_error))**2 + ( phi2d * delta_relative_error / (1+relative_error)**2 )**2)