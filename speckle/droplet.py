
"""
Droplet Algorithms
"""

import numpy as np
import scipy.ndimage.measurements as smt 
import scipy.ndimage.morphology as smf

try:
    from skbeam.core.accumulators.droplet import dropletfind, dropletanal
    _SKBEAM = True
except ImportError as e:
    _SKBEAM = False
    

def dropletize(img, threshold=10.0, dilate=1, return_coms=False):
    """
    A simple and very effective threshold-based droplet algorithm.

    Works only in the case where droplets form disjoint regions (sparse case).
    The algorithm thresholds the image and looks for connected regions above
    the threshold. Each connected region becomes a droplet.

    Parameters
    ----------
    img : np.ndarray
        The two-D image to search for droplets in.

    threhsold : float
        The threshold for the image. Should be between 0 and 1/2 a photon
        in detector gain units.

    dilate : int
        Optionally extend the droplet regions this amount around the thresholded
        region. This lets you be sure you capture all intesity if using
        an aggresive threshold.

    return_coms : bool
        Whether or not to return the droplet positions (centers of mass).

    Returns
    -------
    adus : list of floats
        The summed droplet intensities in detector gain units (ADUs).

    coms : np.ndarray
        The x,y positions of each droplet found.
    """

    bimg = (img > threshold)
    if dilate > 0:
        bimg = smf.binary_dilation(bimg, iterations=dilate)
    limg, numlabels = smt.label(bimg)

    adus = smt.sum(img, labels=limg, index=np.arange(2,numlabels))

    if return_coms:
        coms = np.array(smt.center_of_mass(img, labels=limg,
                                        index=np.arange(2,numlabels))) 
        return adus, coms

    else:
        return adus


def skbeam_dropletize(img, threshold=10.0):
    """
    Scikit-beam's droplet algorithm. This is originally from Mark Sutton.
    """
    if _SKBEAM is False:
        raise ImportError('You need the droplet branch of skbeam installed')
    bimg = (img > threshold).astype(np.int)
    npeaks, limg = dropletfind(bimg) 
    npix, xcen, ycen, adus, idlist = dropletanal(img.astype(np.int), 
                                                 limg, npeaks) 
    return adus, (xcen, ycen)

