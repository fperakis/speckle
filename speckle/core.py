
"""
This file contains the main functions for computing quantities of interest for
speckle analysis from CSPAD images.
"""

import os

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
from scipy.special import gamma, gammaln
from scipy.special import psi as digamma

try:
    import psana
except ImportError as e:
    print ('Could not import psana, proceeding')


def init_psana(run, expt='', config_file=None):
    """
    Convience function to obtain the datastream and epics objects from psana
    """
    
    if config_file == None:
        config_file = os.path.join("/reg/d/psdm/xcs/%s/res/cfg/%s.cfg" (expt, expt))
    if not os.path.exists(config_file):
        raise IOError('Requested psana configuration file does not exist:'
                      ' %s' % config_file)
    
    psana.setConfigFile(config_file)
    psana.setOption('psana.l3t-accept-only',0)
    print "Loading psana config file:    %s" % config_file
    
    ds = psana.DataSource('exp=%s:run=%d' % (expt, run))
    epics = ds.env().epicsStore()
    
    return ds, epics


def autocorrelate_image(cspad_image, window=10):
    """
    Autocorrelate an image to determine the distribution of speckle sizes.
    
    Parameters
    ----------
    cspad_image : np.ndarray
        The image, (32, 185, 388) format
        
    window : int
        The window size to look at, in pixels. Should completely enclose the
        speckle of interest.
        
    Returns
    -------
    acf : np.ndarray
        A (`window`, `window`) shape array containing the autocorrelation
        of the image. Not normalized.
        
    Notes
    -----
    Tests indicate current implementation will run at ~2.4s/image = 0.4 Hz.
    """
    
    if not cspad_image.shape == (32, 185, 388):
        raise ValueError('`cspad_image` incorrect shape. Expected (32, 185, '
                         '388), got %s' % str(cspad_image.shape))
    
    acf = np.zeros((185, 388))
    
    for two_by_one in cspad_image:
        x = two_by_one - two_by_one.mean()
        acf += fftconvolve(x, x[::-1,::-1])
        
    acf /= float( cspad_image.shape[0] )
        
    return acf
    
    
def ADU_to_photons(cspad_image, cuts):
    """
    Given `cuts` that demarkate photon bins in ADU units, transform 
    `cspad_image` from ADU units to photon counts.
    """
    return np.digitize(cspad_image.flatten(), cuts).reshape(cspad_image.shape)

    
def fit_negative_binomial(samples, method='ml'):
    """
    Estimate the parameters of a negative binomial distribution, using either
    a maximum-likelihood fit or an analytic estimate appropriate in the low-
    photon count limit.
    
    Parameters
    ----------
    samples : np.ndarray, int
        An array of the photon counts for each pixel.
    
    method : str, {"ml", "expansion"}
        Which method to use to estimate the contrast.
        
    Returns
    -------
    contrast : float
        The contrast of the samples.
        
    sigma_contrast : float
        The first moment of the parameter estimation function.
    """
    
    k = samples.flatten()
    N = float( len(k) )
    k_bar = np.mean(samples)
    
    if method == 'ml': # use maximium likelihood estimation
        
        def logL_prime(contrast):
            M = 1.0 / contrast
            t1 = -N * (np.log(k_bar/M + 1.0) + digamma(M))
            t2 = np.sum( (k_bar - k)/(k_bar + M) + digamma(k + M) )
            return t1 + t2
       
        try: 
            contrast = optimize.brentq(logL_prime, 1e-6, 1.0)
        except ValueError as e:
            print e
            raise ValueError('log-likelihood function has no maximum given'
                             ' the empirical example provided. Please samp'
                             'le additional points and try again.')
        
        def logL_dbl_prime(contrast):
            M = 1.0 / contrast
            t1 = np.sum( (np.square(k_bar) - k*M) / (M * np.square(k_bar + M)) )
            t2 = - N * digamma(M)
            t3 = np.sum( digamma(k + M) )
            return t1 + t2 + t3
            
        sigma_contrast = logL_dbl_prime(contrast)
        if sigma_contrast < 0.0:
            raise RuntimeError('Maximum likelihood optimization found a local '
                               'minimum instead of maximum! sigma = %s' % sigma_contrast) 
                               
        
    elif method == 'expansion': # use low-order expansion
        # directly from the SI of the paper in the doc string
        p1 = np.sum( k == 1 ) / N
        p2 = np.sum( k == 2 ) / N
        print p1, p2
        contrast = (2.0 * p2 * (1.0 - p1) / np.square(p1)) - 1.0
        
        # this is not quite what they recommend, but it's close...
        # what they recommend is a bit confusing to me atm --TJL
        sigma_contrast = np.power(2.0 * (1.0 + contrast) / N, 0.5) / k_bar
        
        
    else:
        raise ValueError('`method` must be one of {"ml", "expansion"}')
    
    return contrast, sigma_contrast
    
    
def negative_binomial_pdf(k_range, k_bar, contrast):
    """
    
    
    Parameters
    ----------
    
    """
    M = 1.0 / contrast
    norm = np.exp(gammaln(k_range + M) - gammaln(M) - gammaln(k_range+1))
    f1 = np.power(1.0 + M/k_bar, -k_range)
    f2 = np.power(1.0 + k_bar/M, -M)
    return norm * f1 * f2

    
    
