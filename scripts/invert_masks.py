#!/usr/bin/env python

import numpy as np
import glob

fnames = glob.glob(  "/reg/d/psdm/xcs/xcslr0016/scratch/masks/inverted/*.npy")

for f in fnames:
    # shift to new directory
    new_f = f.replace("scratch/masks/inverted/", "scratch/masks/")
    mask = np.load(f).astype(np.bool)
    np.save(new_f, np.logical_not(mask))
    print "inverted %s" % (f.split('/')[-1])


