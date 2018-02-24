#!/usr/bin/env python
import numpy as n
#import EdfFile
import matplotlib.mlab
#from matplotlib.nxutils import points_inside_poly
from matplotlib.path import Path
import pylab as p
import os.path
from numpy.ma import masked_array
from commands import getoutput 
import h5py as h

class Make_mask:

   def __init__(self,data,mask_file,threshold=0,row13=False):
      self.key=[]  
      self.x=0
      self.y=0
      self.xy=[]
      self.xx=[]
      self.yy=[]
      self.data_shape = data.shape
      # cspad 2x2 shape: (2, 185, 388)
      if len(data.shape) == 3:
         lx = data.shape[0]
         ly = data.shape[1]
         lz = data.shape[2]
         data_2d = data.reshape(lx*ly, lz)
         # transpose last 2x1
         data_2d[(lx-1)*ly:lx*ly,:] = data[lx-1,::-1,::-1]
         self.data=data_2d
      else:
         self.data=data
      self.lx,self.ly=p.shape(self.data)
      self.points=[]
      for i in range(self.lx):
          for j in range(self.ly):
           self.points.append([i,j]) 
      self.mask_file=mask_file
      if os.path.exists(self.mask_file) is True:
#         if 'edf' in self.mask_file:
#            mask_f=EdfFile.EdfFile(self.mask_file)
#            self.mymask=mask_f.GetData(0)
#            num_masks=mask_f.GetNumImages()
#            if num_masks==2:
#               self.automask=mask_f.GetData(1)
#               self.anisotropic_mask=0*self.mymask
#            if num_masks==3:
#               self.automask=mask_f.GetData(1)
#               self.anisotropic_mask=mask_f.GetData(2)
#            else:
#               self.automask=0*self.mymask
#               self.anisotropic_mask=0*self.mymask
#            if p.shape(self.mymask)!=p.shape(self.data):
#               self.mymask=n.zeros((self.lx,self.ly))
#         elif 'h5' in self.mask_file:
         if 'h5' in self.mask_file:
            #newfile=self.make_2mask_name()
            self.open_mask()
            self.anisotropic_mask=0*self.mymask
      else:
         self.mymask=n.zeros((self.lx,self.ly))
         self.automask=n.zeros((self.lx,self.ly))
         self.anisotropic_mask=n.zeros((self.lx,self.ly))
      self.old_mymask=self.mymask
      self.old_automask=self.automask
      self.automask[n.where(self.data<threshold)]=1
      print "automatically masking out " + str(int(self.automask.sum())) + " pixels below threshold=%s" % (threshold)
      #this is for masking out row13 of the CSPAD
      if (row13):
         print "automatically masking out row13 of CSPAD"
         col=181
         for i in range(8):
            self.automask[:,col]=1
            col+= 194
      #end of CSPAD part
      palette=p.cm.jet
      palette.set_bad('w',1.0)
      p.rc('image',origin = 'lower')
      p.rc('image',interpolation = 'nearest')
      p.figure(2)
      self.px=p.subplot(111)
      lowest_value_allowed = -0.999999
      self.data[n.where(self.data < lowest_value_allowed)] = lowest_value_allowed
      self.data=p.log(self.data+1)
      self.im=p.imshow(masked_array(self.data,self.mymask+self.automask+self.anisotropic_mask), cmap=palette)
      p.title('Select a ROI. Press m to mask it or u to unmask it. k to save/exit, q to exit without saving')
      self.lc,=self.px.plot((0,0),(0,0),'-+m',linewidth=1,markersize=8,markeredgewidth=1)
      self.lm,=self.px.plot((0,0),(0,0),'-+m',linewidth=1,markersize=8,markeredgewidth=1)
      self.px.set_xlim(0,self.ly)
      self.px.set_ylim(0,self.lx)
      self.colorbar=p.colorbar(self.im,pad=0.01)
      cidb=p.connect('button_press_event',self.on_click)
      cidk=p.connect('key_press_event',self.on_click)
      cidm=p.connect('motion_notify_event',self.on_move)
      p.show()
   def on_click(self,event):
       if not event.inaxes: 
           self.xy=[]
           return
       self.x,self.y=int(event.xdata), int(event.ydata)
       self.key=event.key
       self.xx.append([self.x])
       self.yy.append([self.y])
       self.xy.append([self.y,self.x])
       self.lc.set_data(self.xx,self.yy)
       if self.key=='m': 
           print 'masking'
           self.xx[-1]=self.xx[0]
           self.yy[-1]=self.yy[0]
           self.xy[-1]=self.xy[0]
           previously_masked = self.mymask.sum()
           #ind=p.nonzero(points_inside_poly(self.points,self.xy))
           verts = []
           codes = []
           #print self.xy # these are the masked vertices
           for xy in self.xy:
              verts.append(xy)
              codes.append(Path.LINETO)
           codes[0] = Path.MOVETO
           codes[-1] = Path.CLOSEPOLY # need extra empty element?
           masked_path = Path(verts, codes)
           self.mymask=self.mymask.reshape(self.lx*self.ly,1)
           inds = []
           #icnt = 0
           #print self.points # these are the whole detector array
           for point in self.points:
              #ind = p.nonzero(masked_path.contains_point(point))
              inds.append(masked_path.contains_point(point))
              #icnt += 1
           self.mymask[p.nonzero(inds)]=1
           self.mymask=self.mymask.reshape(self.lx,self.ly)
           #print icnt
           print "masked out %d pixels (%d already masked)" % (self.mymask.sum()-previously_masked, n.sum(inds)+previously_masked-self.mymask.sum())
           datamasked=masked_array(self.data,self.mymask+self.automask+self.anisotropic_mask)
           self.im.set_data(datamasked)
           self.xx=[]
           self.yy=[]
           self.xy=[] 
           self.lc.set_data(self.xx,self.yy)
           self.lm.set_data(self.xx,self.yy)
#           self.im.set_clim(vmax=(2*self.data.mean()))
           self.im.autoscale()
           p.draw()
           self.x=0
           self.y=0 
       if self.key=='u':
           print 'unmasking'
           self.xx[-1]=self.xx[0]
           self.yy[-1]=self.yy[0]
           self.xy[-1]=self.xy[0]
           previously_masked = self.mymask.sum()
           #ind=p.nonzero(points_inside_poly(self.points,self.xy))
           verts = []
           codes = []
           #print self.xy # these are the masked vertices
           for xy in self.xy:
              verts.append(xy)
              codes.append(Path.LINETO)
           codes[0] = Path.MOVETO
           codes[-1] = Path.CLOSEPOLY # need extra empty element?
           masked_path = Path(verts, codes)
           self.mymask=self.mymask.reshape(self.lx*self.ly,1)
           inds = []
           #print self.points # these are the whole detector array
           for point in self.points:
              inds.append(masked_path.contains_point(point))
           self.mymask[p.nonzero(inds)]=0
           self.mymask=self.mymask.reshape(self.lx,self.ly)
           print "ummasked %d pixels (%d already unmasked)" % (previously_masked-self.mymask.sum(), n.sum(inds)-previously_masked+self.mymask.sum())
           datanew=masked_array(self.data,self.mymask+self.automask+self.anisotropic_mask)
           self.im.set_data(datanew)
           self.xx=[]
           self.yy=[]
           self.xy=[]
           self.lc.set_data(self.xx,self.yy)
           self.lm.set_data(self.xx,self.yy)
#           self.im.set_clim(vmax=(2*self.data.mean()))
           self.im.autoscale()
           p.draw()
           self.x=0
           self.y=0

       if self.key=='r':
           print 'unmasking all'
           self.mymask=0*self.mymask
           datanew=masked_array(self.data,self.mymask+self.automask+self.anisotropic_mask)
           self.im.set_data(datanew)
           self.xx=[]
           self.yy=[]
           self.xy=[] 
           self.lc.set_data(self.xx,self.yy)
           self.lm.set_data(self.xx,self.yy)

#           self.im.set_clim(vmax=(2*self.data.mean()))
           self.im.autoscale()
           p.draw()
           self.x=0
           self.y=0 
       if self.key=='k':
          print 'save and exit'
          self.save_mask()
          print 'Mask saved in file:', self.mask_file
#          mask_f=EdfFile.EdfFile(self.mask_file)
#          mask_f.WriteImage({},self.mymask,0)
#          mask_f.WriteImage({},self.automask,1)
#          mask_f.WriteImage({},self.anisotropic_mask,2)
#          del(mask_f)
          p.close()
          return self.mymask+self.automask
       if self.key=='q':
          print 'exit without saving'
          p.close()
          return self.old_mymask+self.old_automask
   
   def on_move(self,event):
       if not event.inaxes: return
       xm,ym=int(event.xdata), int(event.ydata)
       # update the line positions
       if self.x!=0: 
           self.lm.set_data((self.x,xm),(self.y,ym))
           p.draw()
   
   def save_mask(self):
       # save double-mask for additional masking
       newfile=self.make_2mask_name()
       h5file = h.File(newfile,'w')
       datagroup = h5file.create_group("masks")
       dataset = datagroup.create_dataset("user",self.mymask.shape,dtype="float")
       dataset[...] = self.mymask[:,:]
       dataset2 = datagroup.create_dataset("auto",self.mymask.shape,dtype="float")
       dataset2[...] = self.automask[:,:]
       h5file.close()
       # save npy-mask for psana
       newfile=self.make_npy_name()       
       masktot = n.ceil((self.mymask+self.automask)/2)
       # transform 2x2 cspad back to original shape: (2, 185, 388)
       if len(self.data_shape) == 3:
          lx = self.data_shape[0]
          ly = self.data_shape[1]
          lz = self.data_shape[2]
          # transpose last 2x1
          masktot[(lx-1)*ly:lx*ly,:] = masktot[lx*ly:(lx-1)*ly-1:-1,::-1]
          masktot = masktot.reshape(lx, ly, lz)
       n.save(newfile, masktot)
       # invert and save h5-mask for cheetah
       tosave=abs(masktot-1)
       h5file = h.File(self.mask_file,'w')
       datagroup = h5file.create_group("data")
       dataset = datagroup.create_dataset("data",tosave.shape,dtype="int16")
       dataset[...] = tosave[:,:]
       h5file.close()
       print "saved mask with %d pixels masked out (%d automatic, %d manual)" % (masktot.sum(), self.automask.sum(), self.mymask.sum())
   def open_mask(self):
       newfile=self.make_2mask_name()
       h5file = h.File(newfile,'r')
       self.mymask=n.array(h5file.get("masks/user"))
       self.automask=n.array(h5file.get("masks/auto"))
       masktot = n.ceil((self.mymask+self.automask)/2)
       print "opened mask with %d pixels masked out (%d automatic, %d manual)" % (masktot.sum(), self.automask.sum(), self.mymask.sum())
   def make_2mask_name(self):
       dirname,filename=os.path.split(self.mask_file)
       base,ext=filename.split('.')
       newname=base+'_2masks.'+ext
       newfile=os.path.join(dirname,newname)
       return newfile
   def make_npy_name(self):
       dirname,filename=os.path.split(self.mask_file)
       base,ext=filename.split('.')
       newname=base+'.npy'
       newfile=os.path.join(dirname,newname)
       return newfile
