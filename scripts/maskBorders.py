#!/usr/bin/env python                                                                                         
import os
import sys
import string
import numpy as N
import h5py as H
import matplotlib
import matplotlib.pyplot as P
from optparse import OptionParser

# Usage:
# First argument: input HDF5 file containing pixel mask in /data/data
# Second argument: output HDF5 file containing pixel mask with masked borders in /data/data

#inputfile = sys.argv[1]
#outputfile = sys.argv[2]

parser = OptionParser()
parser.add_option("-f", "--file", action="store", type="string", dest="fileName",
                                        help="input mask you wish to mask borders", metavar="FILENAME", default="")
parser.add_option("-o", "--output", action="store", type="string", dest="maskName",
                                        help="output mask you wish to save (default: adds 'borders' before suffix)", metavar="MASKNAME", default="")
parser.add_option("-r", "--row", action="store", type="int", dest="maskRows",
                                        help="number of rows to mask for each border (default: 1)", metavar="NROWS", default="1")

(options, args) = parser.parse_args()

inputfile = options.fileName
if options.maskName == "":
        base,ext = options.fileName.split('.')
        outputfile = base + "+borders." + ext
else:
        outputfile = options.maskName
#write_dir = os.getcwd()
write_dir = '/reg/d/psdm/xcs/xcslr0016/scratch/masks/inverted/'

if "epix" in inputfile.lower():
        ASICS = 2
        ROWS = 704/ASICS
        COLS = 768/ASICS
        RAW_DATA_LENGTH = [ROWS*ASICS,COLS*ASICS]
elif "cspad" in inputfile.lower():
        ASICS = 2
        ROWS = 185
        COLS = 194
        RAW_DATA_LENGTH = [ROWS*ASICS,COLS*ASICS]
        #RAW_DATA_LENGTH = [ROWS*ASICS,COLS]
else:
        "Unknown detector type (from filename): %s" % inputfile
        sys.exit(1)

# Open input mask
if inputfile.endswith('.h5'):
        f = H.File(write_dir+inputfile,"r")
        mask = N.array(f['/data/data'])
        f.close()
elif inputfile.endswith('.npy'):
        mask = N.load(write_dir+inputfile)
        if "cspad" in inputfile.lower():
                # reshape mask
                if len(mask.shape) == 3:
                        lx = mask.shape[0]
                        ly = mask.shape[1]
                        lz = mask.shape[2]
                        mask2d = mask.reshape(lx*ly, lz)
                        # transpose last 2x1
                        mask2d[(lx-1)*ly:lx*ly,:] = mask[lx-1,::-1,::-1]
                        mask = mask2d
                else:
                        lx = 0
                        ly = 0
                        lz = 0
                #mask = mask.reshape(ASICS*ROWS, ASICS*COLS)
                #mask = N.vstack(mask)
        # invert mask
        mask = N.abs(mask - 1)
else:
        print "Unknown file format: %s" % inputfile
        sys.exit(1)

masked = N.where(mask < 1)
print "%s contains %s masked pixels." % (inputfile, len(mask[masked]))

########################################################
# Imaging class copied from Ingrid Ofte's pyana_misc code
########################################################
class img_class (object):
	def __init__(self, inarr, filename):
		self.inarr = inarr*(inarr>0)
		self.filename = filename
		self.cmax = self.inarr.max()
		self.cmin = self.inarr.min()
	
	def on_keypress(self,event):
		if event.key == 'p':
			if not os.path.exists(write_dir):
				os.mkdir(write_dir)
			pngtag = write_dir + "/%s.png" % (self.filename)	
			print "saving image as " + pngtag 
			P.savefig(pngtag)
		if event.key == 'r':
			colmin, colmax = self.orglims
			P.clim(colmin, colmax)
			P.draw()


	def on_click(self, event):
		if event.inaxes:
			lims = self.axes.get_clim()
			colmin = lims[0]
			colmax = lims[1]
			range = colmax - colmin
			value = colmin + event.ydata * range
			if event.button is 1 :
				if value > colmin and value < colmax :
					colmin = value
			elif event.button is 2 :
				colmin, colmax = self.orglims
			elif event.button is 3 :
				if value > colmin and value < colmax:
					colmax = value
			P.clim(colmin, colmax)
			P.draw()
				

	def draw_img(self):
		fig = P.figure()
		cid1 = fig.canvas.mpl_connect('key_press_event', self.on_keypress)
		cid2 = fig.canvas.mpl_connect('button_press_event', self.on_click)
		canvas = fig.add_subplot(111)
		canvas.set_title(self.filename)
		self.axes = P.imshow(self.inarr, origin = 'lower', vmax = self.cmax, interpolation='nearest')
		self.colbar = P.colorbar(self.axes, pad=0.01)
		self.orglims = self.axes.get_clim()
		P.show()


#print "Right-click on colorbar to set maximum scale."
#print "Left-click on colorbar to set minimum scale."
#print "Center-click on colorbar (or press 'r') to reset color scale."
#print "Interactive controls for zooming at the bottom of figure screen (zooming..etc)."
#print "Press 'p' to save PNG of image (with the current colorscales) in the appropriately named folder."
#print "Hit Ctl-\ or close all windows (Alt-F4) to terminate viewing program."

maskImg = img_class(mask,inputfile)
maskImg.draw_img()

# Create output array of type 16-bit integer with masked borders
outputmask = mask.astype(N.int16)
for asic in N.arange(ASICS):
	for n in N.arange(options.maskRows):
		outputmask[asic*ROWS+n,:] = 0
		outputmask[asic*ROWS+ROWS-1-n,:] = 0
		outputmask[:,asic*COLS+n] = 0
		outputmask[:,asic*COLS+COLS-1-n] = 0

if outputfile.endswith('.h5'):
	# Save output array to HDF5 file
	h5output = H.File(write_dir+outputfile,'w')
	datagroup = h5output.create_group("data")
	dataset = datagroup.create_dataset("data",RAW_DATA_LENGTH,dtype="int16")
	dataset[...] = outputmask[:,:]
	h5output.close()
elif outputfile.endswith('.npy'):
        # Save output array as numpy file
        npymask = N.abs(outputmask - 1)
        if "cspad" in inputfile.lower():
                # reshape mask
                #npymask = npymask.reshape(ASICS, ROWS, ASICS*COLS)
                if lx != 0:
                        # transpose last 2x1
                        npymask[(lx-1)*ly:lx*ly,:] = npymask[lx*ly:(lx-1)*ly-1:-1,::-1]
                        npymask = npymask.reshape(lx, ly, lz)

        N.save(write_dir+outputfile, npymask)
else:
        print "Unknown file format: %s" % outputfile
        sys.exit(1)

maskedo = N.where(outputmask < 1)
print "Saved new mask as %s containing %s masked pixels." % (outputfile, len(outputmask[maskedo]))

outputmaskImg = img_class(outputmask, outputfile)
outputmaskImg.draw_img()
