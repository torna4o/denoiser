import numpy as np
import matplotlib.pyplot as plt


# Generate Gaussian noise

def add_gauss(img, mean=0, std=1, speckle = False):

    """
    This function adds a gaussian noise to a numpy.ndarray.
    It takes a numpy.ndarray as input and returns gaussian added array
    and gaussian noise itself separately.
    
    "img" = a numpy.ndarray format image data in 2D
    "mean" = mean of the gaussian noise array to be generated.
    "std" = standard deviation of the gaussian noise array to be generated.
    """

    gauss = np.random.normal((img.shape[0],img.shape[1]), min=mean - std,max=mean + std)

    if speckle==False:
        img_gauss = img + gauss
    elif speckle==True:
        img_gauss = img + img * gauss
    return img_gauss, gauss

def add_poisson(img, lam):
    
    """
    
    This function adds a Poisson noise to a numpy.ndarray.
    It takes a numpy.ndarray as input and returns Poisson added array
    and Poisson noise itself separately.

    "img" = a numpy.ndarray format image data in 2D   
    "lam" = lambda of Poisson distribution.

    """

    pois = np.random.poisson(lam, (img.shape[0], img.shape[1]))
    img_pois = img + pois
    
    return img_pois, pois

def add_band_spec_noise(min_freq, max_freq, samples=1024, samplerate=1):
    """
    This code is mainly taken from this page:
    https://stackoverflow.com/a/35091295/12656842
    
    The aim is creating a specific frequency band noise
    via numpy.fft functions and several other numpy functions
    
    min_freq = minimum frequency of the specific noise
    max_freq = maximum frequency of the specific noise
    samples = effectively this is the size of the noise
    samplerate = this adjusts the given frequency rate and resulting 
        frequencies in the noise
    
    Example:
    
    import noisy as ns
    # following produces a 512-length sinusoidal noise with 25 length cycles
    ns.band_limited_noise(25,25, samples=512, samplerate=1/512)
    
    """
    
    # Generating a real noise from a DFT routine    
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    
    f = np.zeros(samples)
    # Locations of the array corresponding to the specified frequency range
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    # Preparation to the phase addition to the array
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    # Remined, indexing works like [start:end:step]
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real    
    

def saltpep(img, multi, threshold=0.99):

    """
    Adds salt & pepper noise using NumPy package
    
    img = image data in numpy.ndarray that the noise will be ingested
    threshold = cutoff for making salt and pepper noise, 1 will result
        in full salt, 0 full pepper noise
    multi = this is a multiplier, in case one needs different numbers than 
        1 salt noise generation
    """
    base = np.random.rand(img.shape[0], img.shape[1])
    base[base >= threshold] = 1
    base[base < threshold] = 0
    base *= multi
    return np.add(img, base)
