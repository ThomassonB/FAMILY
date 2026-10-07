def estimated_f(f_up, f_low):
    return (f_up + f_low)/2, (f_up - f_low)/2

def phi2D(fractality):
    return np.log(fractality) / np.log(2)

def delta_phi2D(fractality, delta_fractality):
    return delta_fractality / (fractality * np.log(2))

def phi3D(phi2d, relative_error):
    return phi2d / (1 + relative_error)

def delta_phi3D(phi2d, delta_phi2d, relative_error, delta_relative_error):
    return np.sqrt( (delta_phi2d / (1+f))**2 + ( phi2d * delta_relative_error / (1+f)**2 )**2)