#!/usr/bin/env python

# Usage:
# In this directory, type:
#    ./maskAvIm.py -rxxxx 
# For details, type 
#	 python maskAvIm --help
# where rxxxx is the run number 
# By default, this script looks into the h5 files that are in the appropriate rxxxx directory
#

import os
import sys
import string
import re
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--file", action="store", type="string", dest="fileName", 
					help="input file you wish to mask", metavar="FILENAME", default="")
parser.add_option("-m", "--mask", action="store", type="string", dest="maskName",
					help="output mask you wish to save", metavar="MASKNAME", default="mask.h5")
parser.add_option("-t", "--threshold", action="store", type="float", dest="threshold", 
					help="threshold (ADU) below which automasking is performed (default: 0)", metavar="THRESHOLD", default="0")
parser.add_option("-o", "--offset", action="store", type="float", dest="offset", 
					help="offset (ADU) to shift data so that all pixels are positive (default: 0)", metavar="OFFSET", default="0")
parser.add_option("-e", "--epix", action="store_true", dest="epix", default=False, help="process epix")
parser.add_option("-c", "--cspad", action="store_true", dest="cspad", default=False, help="process cspad")

(options, args) = parser.parse_args()

import numpy as N
import h5py as H

import matplotlib
import matplotlib.pyplot as P

import pylab as p
import pickle

from Make_mask import Make_mask
from numpy.ma import masked_array

########################################################
# Edit this variable accordingly
# Files are read for source_dir/runtag and
# written to write_dir/runtag.
# Be careful of the trailing "/"; 
# ensure you have the necessary read/write permissions.
########################################################
#source_dir = os.getcwd()+'/'
source_dir = '/reg/d/psdm/xcs/xcslr0016/scratch/combined/'
#write_dir = source_dir
write_dir = '/reg/d/psdm/xcs/xcslr0016/scratch/masks/inverted/'

filename = options.fileName
maskname = options.maskName
print source_dir+filename
if filename.endswith('.h5') or filename.endswith('.tbl'):
        f = H.File(source_dir+filename,"r")
        if options.cspad:
                print "Reading CSPAD data.."
                if "mask" in filename:
                        d = N.array(f['/data/data'])
                else:
                        #d = N.vstack(N.array(f['/cspad/sum']))
                        d = N.array(f['/cspad/sum'])
        if options.epix:
                print "Reading ePIX data.."
                e = []
                if "mask" in filename:
                        e.append(N.array(f['/data/data']))
                else:
                        for i in range(1,5):
                                e.append(N.array(f['/epix_sum/epix_%d' % i]))
        f.close()
elif filename.endswith('.npy'):
        d = N.load(source_dir+filename)
else:
        print "Unknown file format: %s" % filename
        sys.exit(1)

if not maskname.endswith('.h5'):
        base,ext = maskname.split('.')
        maskname = base + '.h5'
        #maskname = re.sub('.npy', '.h5', maskname)
        print "Changed maskname from %s to : %s" % (base + ext, maskname)

if options.cspad:
        print "Making mask for CSPAD.."
        mask_cspad=Make_mask(d+options.offset,re.sub('.h5', '_cspad.h5', write_dir+maskname),options.threshold)
if options.epix:
        print "Making mask for ePIX.."
        mask_epix = []
        for i in range(1,len(e)+1):
                mask_epix.append(Make_mask(e[i-1]+options.offset,re.sub('.h5', '_epix%d.h5' % i, write_dir+maskname),options.threshold))

########################################################
# Imaging class copied from Ingrid Ofte's pyana_misc code
########################################################
class img_class (object):
	def __init__(self, inarr, filename):
		self.inarr = inarr*(inarr>0)
		self.filename = filename
		self.cmax = 0.1*self.inarr.max()
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
		P.rc('image',origin='lower')
		self.axes = P.imshow(self.inarr, vmax = self.cmax)
		self.colbar = P.colorbar(self.axes, pad=0.01)
		self.orglims = self.axes.get_clim()
		P.show()

#print "Right-click on colorbar to set maximum scale."
#print "Left-click on colorbar to set minimum scale."
#print "Center-click on colorbar (or press 'r') to reset color scale."
#print "Interactive controls for zooming at the bottom of figure screen (zooming..etc)."
#print "Press 'p' to save PNG of image (with the current colorscales) in the appropriately named folder."
#print "Hit Ctl-\ or close all windows (Alt-F4) to terminate viewing program."

#currImg = img_class(d, filename)
#currImg.draw_img()
