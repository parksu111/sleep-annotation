#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:47:50 2017
@author: Franz Weber
@Additions made by: Sung-Ho Park

To run program, type
python sleep_annotation_qt.py "path/to/recording_folder"
"""

import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import pyqtgraph as pg
import numpy as np
import scipy.io as so
import os
import re
import h5py
import sleepy


def get_cycles(ppath, name):
	"""
	extract the time points where dark/light periods start and end
	"""
	act_dur = sleepy.get_infoparam(os.path.join(ppath, name, 'info.txt'), 'actual_duration')

	time_param = sleepy.get_infoparam(os.path.join(ppath, name, 'info.txt'), 'time')
	if len(time_param) == 0 or len(act_dur) == 0:
		return {'light': [(0,0)], 'dark': [(0,0)]}
	
	hour, mi, sec = [int(i) for i in re.split(':', time_param[0])]
	#a = sleepy.get_infoparam(os.path.join(ppath, name, 'info.txt'), 'actual_duration')[0]
	a,b,c = [int(i[0:-1]) for i in re.split(':', act_dur[0])]
	total_dur = a*3600 + b*60 + c
	
	# number of light/dark switches
	nswitch = int(np.floor(total_dur / (12*3600)))
	switch_points = [0]
	cycle = {'light': [], 'dark':[]}
		
	if hour >= 7 and hour < 19:
		# recording starts during light cycle
		a = 19*3600 - (hour*3600+mi*60+sec)
		for j in range(nswitch):
			switch_points.append(a+j*12*3600)
		for j in range(1, nswitch, 2):
			cycle['dark'].append(switch_points[j:j+2])
		for j in range(0, nswitch, 2):
			cycle['light'].append(switch_points[j:j+2])
		
	else:
		# recording starts during dark cycle
		a = 0
		if hour < 24:
			a = 24 - (hour*3600+mi*60+sec) + 7*3600
		else:
			a = 7*3600 - (hour*3600+mi*60+sec)
		for j in range(nswitch):
			switch_points.append(a+j*12*3600)
		for j in range(0, nswitch, 2):
			cycle['dark'].append(switch_points[j:j+2])
		for j in range(1, nswitch, 2):
			cycle['light'].append(switch_points[j:j+2])
		
	return cycle



def load_stateidx(ppath, name):
	""" load the sleep state file of recording ppath/name
	@RETURN:
	M     -      sequency of sleep states
	"""   
	file = os.path.join(ppath, name, 'remidx_' + name + '.txt')
	
	f = open(file, newline=None)    
	lines = f.readlines()
	f.close()
	
	n = 0
	for l in lines:
		if re.match('\d', l):
			n = n+1
			
	M = np.zeros(n)
	K = np.zeros(n)
	
	i = 0
	for l in lines :
		
		if re.search('^\s+$', l) :
			continue
		if re.search('\s*#', l) :
			continue
		
		if re.match('\d+\s+-?\d+', l) :
			a = re.split('\s+', l)
			M[i] = int(a[0])
			K[i] = int(a[1])
			i += 1
			
	return M,K

def load_stateprob(ppath, name):
	""" load the sleep state file of recording ppath/name
	@RETURN:
	M     -      sequency of sleep states
	"""   
	file = os.path.join(ppath, name, 'remprob_' + name + '.txt')
	
	f = open(file, newline=None)    
	lines = f.readlines()
	f.close()
	
	n = 0
	for l in lines:
		if re.match('\d', l):
			n = n+1
			
	M = np.zeros(n)
	K = np.zeros(n)
	
	i = 0
	for l in lines :
		
		if re.search('^\s+$', l) :
			continue
		if re.search('\s*#', l) :
			continue
		
		if re.match('\d+\s+-?\d+', l) :
			a = re.split('\s+', l)
			M[i] = int(a[0])
			K[i] = int(a[1])
			i += 1
			
	return M,K



def rewrite_remidx(M, K, ppath, name):
	"""
	rewrite_remidx(idx, states, ppath, name)
	replace the indices idx in the remidx file of recording name
	with the assignment given in states
	"""
	outfile = os.path.join(ppath, name, 'remidx_' + name + '.txt')

	f = open(outfile, 'w')
	s = ["%d\t%d\n" % (i,j) for (i,j) in zip(M[0,:],K)]
	f.writelines(s)
	f.close()

def rewrite_stateprob(M,K,ppath,name):
	outfile = os.path.join(ppath, name, 'stateprob_'+name+'.txt')
	f = open(outfile, 'w')
	s = ["%d\t%d\n" % (i,j) for (i,j) in zip(M[0,:],K)]
	f.writelines(s)
	f.close()



def get_snr(ppath, name) :
	"""
	read and return SR from file $ppath/$name/info.txt 
	"""
	fid = open(os.path.join(ppath, name, 'info.txt'), newline=None)    
	lines = fid.readlines()
	fid.close()
	values = []
	for l in lines :
		a = re.search("^" + 'SR' + ":" + "\s+(.*)", l)
		if a :
			values.append(a.group(1))            
	return float(values[0])



def laser_start_end(laser, SR=1525.88, intval=5) :
	"""laser_start_end(laser, SR=TDT_sampling_rate, intval=5 [s]) ...
	print start and end index of laser stimulation periods\
	returns the tuple (istart, iend), both indices are inclusive,\
	i.e. part of the sequence
	
	laser    -    laser, vector of 0s and 1s
	intval   -    min time interval [s] between two laser sequences
	"""
	idx = np.nonzero(laser > 0.1)[0]
	if len(idx) == 0 :
		return ([], [])
	
	idx2 = np.nonzero(np.diff(idx)*(1./SR) > intval)[0]
	istart = np.hstack([idx[0], idx[idx2+1]])
	iend   = np.hstack([idx[idx2], idx[-1]])    
	
	return (istart, iend)


def load_laser(ppath, name) :
	"""load laser from ppath, name ...
	@RETURN: @laser, sequency of 0's and 1's """ 
	file = os.path.join(ppath, name, 'laser_'+name+'.mat')
	try:
		laser = np.array( h5py.File(file,'r').get('laser') )
	except:
		laser = so.loadmat(file)['laser']
	return np.squeeze(laser)


def load_trigger(ppath, name) :
	"""load sleep-state detection signal
	recorded during closed loop stimulation
	from ppath, name ...
	@RETURN: @laser, sequency of 0's and 1's """
	file = os.path.join(ppath, name, 'rem_trig_'+name+'.mat')
	try:
		triggered = np.array( h5py.File(file,'r').get('rem_trig') )
	except:
		triggered = so.loadmat(file)['rem_trig']
	return np.squeeze(triggered)


class Second(QtWidgets.QMainWindow):
	def __init__(self, parent=None):
		super(Second, self).__init__(parent)


# Object holding the GUI window
class MainWindow(QtWidgets.QMainWindow):
	def __init__(self, ppath, name):
		QtWidgets.QMainWindow.__init__(self)
		#super(MainWindow, self).__init__()
		
		self.index = 10
		self.ppath = ppath
		self.name  = name
		self.pcollect_index = False
		self.index_list = [self.index]
		self.tscale = 1
		self.tunit = 's'
		# show EEG1 or EEG2?
		self.peeg2 = 0
		# show EMG1 or EMG2?
		self.pemg2 = 0
		# show laser on raw EEG?
		self.pplot_laser = False
		# status variable for marken breaks in recording
		self.pbreak = False
		self.break_index = np.zeros((2,), dtype='int')
		self.setGeometry( QtCore.QRect(100, 100 , 2000, 1000 ))
	 
		#--------------------------------------------------- Create a plot item;
		#-------- This is essentially your graph paper where your graph is drawn
		#self.graph = pg.PlotItem()
	   
		#------------- Put your data inside PlotItem to get your actual plotting
		#self.graph.plot( self.dataPlot)
	   
		#------------------------- You are going to show your graph paper inside
		#---------------------------------- the GraphicsView widget of Pyqtgraph
		#self.view = pg.GraphicsView()
	   
		#--------------------------- if you are going to use multiple plot items
		#--------- in the same graphics view get layout and add items one by one
		#--------- for that uncomment bellow codes (1) ,(2) ,(3) and comment (4)
		#---------- self.lay  = pg.GraphicsLayout()                     #----(1)
		#---------- self.lay.addItem(self.graph)                        #----(2)
		#---------- self.view.setCentralItem(self.lay)                  #----(3)
 
		#-------------------------------- If you display only one graph use this
		#self.view.setCentralItem(self.graph)                #----(4)
	   
		#---------------------------------------- Since we are using QMainWindow
		#--------------------------- put our view as CentralWidget in MainWindow
		#self.setCentralWidget(self.view)
		
		# draw in all data for the first time
		self.show()
		self.load_recording()
		self.gen_layout()
		self.plot_session()
		self.plot_brainstate()
		self.plot_eeg()
		self.plot_treck()
		
		# initial setup for live EEG, EMG
		self.graph_eeg.setRange(yRange=(-500, 500),  padding=None)
		self.graph_emg.setRange(yRange=(-500, 500),  padding=None)
		
		
	def gen_layout(self):
		# live EEG inspection and Annotation
		self.graph_eeg = pg.PlotItem()    
		self.graph_emg = pg.PlotItem()
		self.graph_ann = pg.PlotItem()
		
		self.view = pg.GraphicsView()
		self.lay  = pg.GraphicsLayout()             
		self.lay_brainstate  = self.lay.addLayout() 
		
		#what's annotated, laser, and current visited timepoint
		self.graph_treck = pg.PlotItem()   
		self.lay_brainstate.addItem(self.graph_treck)
		self.lay_brainstate.nextRow()

		# color-coded accuracy
		#self.graph_brainprob = self.lay_brainstate.addPlot()
		#self.image_brainprob = pg.ImageItem()
		#self.graph_brainprob.addItem(self.image_brainprob)
		#self.lay_brainstate.nextRow()

		# color-coded brainstate
		self.graph_brainstate = self.lay_brainstate.addPlot()
		self.image_brainstate = pg.ImageItem() 
		self.graph_brainstate.addItem(self.image_brainstate)
		self.lay_brainstate.nextRow()
		
		# add whole spectrogram
		# self.graph_spectrum contains the image image_spectrum
		self.graph_spectrum = self.lay_brainstate.addPlot()
		self.image_spectrum = pg.ImageItem()     
		#self.image_spectrum = pg.ImageView()
		self.graph_spectrum.addItem(self.image_spectrum)
		self.lay_brainstate.nextRow()
		
		# EMG Amplitude
		self.graph_emgampl = self.lay_brainstate.addPlot()          
		self.lay.nextRow()        
		
		# Add live EEG and EMG     
		self.lay.nextRow()        
		self.lay.addItem(self.graph_eeg) 
		self.lay.nextRow()
		self.lay.addItem(self.graph_emg)
		self.lay.nextRow()

		self.lay_ann = self.lay.addLayout()        
		# add live spectrum        
		self.graph_fft = self.lay_ann.addPlot()
		self.image_fft = pg.ImageItem()        
		self.graph_fft.addItem(self.image_fft)
		self.lay_ann.nextRow()

		# add annotation
		self.lay_ann.addItem(self.graph_ann)

		# explanation how layout in QtMainWindow is organized:
		# https://srinikom.github.io/pyside-docs/PySide/QtGui/QMainWindow.html
		self.view.setCentralItem(self.lay)              
		self.setCentralWidget(self.view)

		#colormap for hypnogram
		pos = np.linspace(0, 1, 8)
		color = np.array(
			[[0, 0, 0, 200], [0, 255, 255, 200], [150, 0, 255, 200], [150, 150, 150, 200], [0,255,43,200], [255,20,20,200], [66,86,219,200], [255,255,0,200]], dtype=np.ubyte)
		cmap = pg.ColorMap(pos, color)
		self.lut_brainstate = cmap.getLookupTable(0.0, 1.0, 8)
		
		#colormap for brainprob
		color2 = np.array([[150, 150, 150, 200], [255, 0, 0,  200], [0,0,0, 200]], dtype=np.ubyte)
		pos2 = np.linspace(0, 1, color2.shape[0])
		cmap2 = pg.ColorMap(pos2, color2)
		self.lut_brainprob = cmap2.getLookupTable(0, 1, color2.shape[0])

		#colormap for spectogram
		pos = np.array([0., 0.05, .2, .4, .6, .9])
		color = np.array([[0, 0, 0, 255], [0,0,128,255], [0,255,0,255], [255,255,0, 255], (255,165,0,255), (255,0,0, 255)], dtype=np.ubyte)
		cmap = pg.ColorMap(pos, color)
		self.lut_spectrum = cmap.getLookupTable(0.0, 1.0, 256)



	def plot_session(self, scale=1, scale_unit = 's'):
		"""
		plot spectrogram and EMG amplitude for whole session
		"""
		
		# clear plot and then reload ImageItem        
		self.graph_spectrum.clear()
		self.image_spectrum = pg.ImageItem() 
		self.graph_spectrum.addItem(self.image_spectrum)
		
		# scale image to seconds, minutes or hours        
		self.image_spectrum.setImage(self.eeg_spec[0:self.ifreq,:].T)
		#self.image_spectrum.scale(self.fdt*scale, 1.0*self.fdx)
		tr1 = QtGui.QTransform()
		tr1.scale(self.fdt*scale, 1.0*self.fdx)
		self.image_spectrum.setTransform(tr1)
		
		# mousescroll only allowed along x axis
		self.graph_spectrum.vb.setMouseEnabled(x=True, y=False)
		limits = {'xMin': -1*self.fdt*scale, 'xMax': self.ftime[-1]*scale, 'yMin': 0, 'yMax': self.freq[self.ifreq]}
		self.graph_spectrum.vb.setLimits(**limits)
		
		# label for y-axis
		ax = self.graph_spectrum.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('Freq', units='Hz', **labelStyle)
		ax.setTicks([[(0, '0'), (10, '10'), (20, '20')]])
		
		# colormap
		self.image_spectrum.setLookupTable(self.lut_spectrum)
		# link graph with self.graph_brainstate
		self.graph_spectrum.setXLink(self.graph_brainstate.vb)
		
		# plot EMG amplitude
		self.graph_emgampl.clear()
		self.graph_emgampl.plot(self.ftime*scale+scale*self.fdt/2.0, self.EMGAmpl)
		self.graph_emgampl.vb.setMouseEnabled(x=False, y=True)
		self.graph_emgampl.setXLink(self.graph_spectrum.vb)
		limits = {'xMin': 0, 'xMax': self.ftime[-1]*scale}
		self.graph_emgampl.vb.setLimits(**limits)
		# y-axis
		ax = self.graph_emgampl.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('EMG Ampl.', units='V', **labelStyle)
		# x-axis        
		ax = self.graph_emgampl.getAxis(name='bottom')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('Time', units=scale_unit, **labelStyle)



	def plot_brainstate(self, scale=1):
		'''
		self.graph_brainprob.clear()
		self.image_brainprob = pg.ImageItem()
		self.graph_brainprob.addItem(self.image_brainprob)

		self.image_brainprob.setImage(self.M2.T)
		self.image_brainprob.scale(self.fdt*scale,1)
		self.graph_brainprob.vb.setMouseEnabled(x=True, y=False)
		self.graph_brainprob.setXLink(self.graph_spectrum.vb)
		limits = {'xMin': -1*self.fdt*scale, 'xMax': self.ftime[-1]*scale, 'yMin': 0, 'yMax': 1}
		self.graph_brainprob.vb.setLimits(**limits)

		ax2 = self.graph_brainprob.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax2.setLabel('Prob', units='', **labelStyle)
		ax2.setTicks([[(0,''),(1,'')]])

		ax2 = self.graph_brainprob.getAxis(name='bottom')
		ax2.setTicks([[]])

		self.image_brainprob.setLookupTable(self.lut_brainprob)
		self.image_brainprob.setLevels([0,2])
		'''

		# clear plot and then reload ImageItem        
		self.graph_brainstate.clear()
		self.image_brainstate = pg.ImageItem() 
		self.graph_brainstate.addItem(self.image_brainstate)
		
		# set image        
		self.image_brainstate.setImage(self.M.T)
		tr2 = QtGui.QTransform()
		tr2.scale(self.fdt*scale,1)
		self.image_brainstate.setTransform(tr2)
		#self.image_brainstate.scale(self.fdt*scale,1)
		self.graph_brainstate.vb.setMouseEnabled(x=True, y=False)
		self.graph_brainstate.setXLink(self.graph_spectrum.vb)
		limits = {'xMin': -1*self.fdt*scale, 'xMax': self.ftime[-1]*scale, 'yMin': 0, 'yMax': 1}
		self.graph_brainstate.vb.setLimits(**limits)

		ax = self.graph_brainstate.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('Brainstate', units='', **labelStyle)
		ax.setTicks([[(0, ''), (1, '')]])
		
		ax = self.graph_brainstate.getAxis(name='bottom')
		ax.setTicks([[]])
		
		self.image_brainstate.setLookupTable(self.lut_brainstate)
		#self.image_brainstate.setLevels([1, 5])
		self.image_brainstate.setLevels([0, 7])


	def plot_treck(self, scale=1):     
		"""
		plot laser, currently visited state and annotated states
		"""
		self.graph_treck.clear()
		self.graph_treck.plot(self.ftime*scale, self.K*0.5, pen=(150,150,150))
		if self.pplot_laser:
			self.graph_treck.plot(self.ftime*scale, self.laser, pen=(0,0,255))

		self.graph_treck.setXLink(self.graph_spectrum.vb)
						
		ax = self.graph_treck.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('Laser', units='', **labelStyle)
		ax.setTicks([[(0, ''), (1, '')]])
		limits = {'xMin': -1*self.fdt*scale, 'xMax': self.ftime[-1]*scale, 'yMin': -1.1, 'yMax': 1.1}
		self.graph_treck.vb.setLimits(**limits)
		# remove ticks on x-axis
		ax = self.graph_treck.getAxis(name='bottom')
		ax.setTicks([[]])

		# plot supplmental signal; for example, rem-online detection
		if self.psuppl:
			self.graph_treck.plot(self.ftime * scale, self.suppl_treck * 0.3, pen=(255, 150, 150))

		self.graph_treck.vb.setMouseEnabled(x=True, y=False)

		# plot dark cycles
		self.graph_treck.plot(self.ftime, np.zeros((self.ftime.shape[0],)), pen=pg.mkPen(width=8, color='w'))
		for d in self.dark_cycle:
			a = int(d[0]/self.fdt)
			b = int(d[1]/self.fdt)
			self.graph_treck.plot(self.ftime[a:b+1]*scale, np.zeros((b-a+1,)), pen=pg.mkPen(width=8, color=(100, 100, 100)))
		
		# plot currently annotated point
		self.graph_treck.plot([self.ftime[self.index]*scale + 0.5*self.fdt*scale], [0.0], pen=(0, 0, 0), symbolPen=(255, 0, 0), symbolBrush=(255, 0, 0),
						symbolSize=5)



	def plot_eeg(self):
		"""
		plot current EEG, EMG, Spectrogram and Brainstate
		Note: the EEG that goes into the calculation of a single spectrogram time bin
		corresponds to the 5s window between the two "blue" points left and right
		to the center (blue) point.
		"""
		
		# the time as calculated by calculate_spectrum.m starts with '0'
		# the first 2.5s bin is calculated based on 5s window. I think the "true" 
		# timepoint should therefore be the center of this window, i.e. 2.5s.
		# That's why I add self.fdt (=2.5s); 
		timepoint = self.ftime[self.index]+self.fdt
		
		self.twin = 2*self.fdt
		
		n = int(np.round((self.twin+self.fdt/2)/self.dt))
		i = int(np.round(timepoint / self.dt))
		ii = np.arange(i-n, i+n+1)
		t = np.arange(timepoint-n*self.dt, timepoint+n*self.dt+self.dt/2, self.dt)
		
		# EEG
		self.graph_eeg.clear()
		self.graph_eeg.setRange(xRange=(t[0],t[-1]), padding=None)
		self.graph_eeg.plot(t,self.EEG[ii]) 
		ax = self.graph_eeg.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('EEG' + ' ' + str(self.eeg_pointer+1), units='V', **labelStyle)
		self.graph_eeg.vb.setMouseEnabled(x=False, y=True)
		ax = self.graph_eeg.getAxis(name='bottom')
		ax.setTicks([[]])                   
		
		# plot laser
		if self.pplot_laser == True:
			self.graph_eeg.plot(t, self.laser_raw[ii]*self.eeg_amp*2, pen=(0,0,255))        
		
		# EMG
		self.graph_emg.clear()
		self.graph_emg.setRange(xRange=(t[0],t[-1]), padding=None)
		self.graph_emg.plot(t,self.EMG[ii])     
		ax = self.graph_emg.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('EMG' + ' ' + str(self.emg_pointer+1), units='V', **labelStyle)
		
		ax = self.graph_emg.getAxis(name='bottom')
		#labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		#ax.setLabel('Time', units='s', **labelStyle)
		#self.graph_emg.setXLink(vbox_eeg)
		self.graph_emg.vb.setMouseEnabled(x=False, y=True)
		
		# indices for life spectrogram and brain state
		n = int(self.twin/self.fdt);
		i = self.index;
		ii = list(range(i-n,i+n+1))
			
		# Live spectrogram
		# clear image
		self.graph_fft.clear()
		self.image_fft = pg.ImageItem() 
		self.graph_fft.addItem(self.image_fft)
		# set image
		self.image_fft.setImage(self.eeg_spec[0:self.ifreq,ii].T)
		tr3 = QtGui.QTransform()
		tr3.scale(1,1.0*self.fdx)
		self.image_fft.setTransform(tr3)
		#self.image_fft.scale(1, 1.0*self.fdx) 
		# set y-axis
		ax = self.graph_fft.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('Freq', units='Hz', **labelStyle)
		ax.setTicks([[(0, '0'), (10, '10'), (20, '20')]])
		limits = {'yMin': 0, 'yMax': self.freq[self.ifreq]}
		
		self.graph_fft.vb.setLimits(**limits)
		# set x-axis
		ax = self.graph_fft.getAxis(name='bottom')
		ax.setTicks([[]])   
		# set colormap and range
		self.image_fft.setLookupTable(self.lut_spectrum)
		self.graph_fft.vb.setMouseEnabled(x=False, y=False)
		self.image_fft.setLevels((0, self.color_max))

		# Sleep Stage annotation
		self.graph_ann.clear()
		# PlotItem inherits all functions from ViewBox; setRange is a method of ViewBox
		self.graph_ann.setRange(yRange=(1, 3), xRange=(0,5), padding=None)
		self.graph_ann.plot(np.arange(0,5)+0.5, self.M[0,ii], name = 'Ann', pen=(255,0,0), symbolPen='w')
		ax = self.graph_ann.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'} 
		ax.setLabel('State', units='', **labelStyle)       
		ax.setTicks([[(1, 'R'), (2, 'W'), (3, 'S')]])

		ax = self.graph_ann.getAxis(name='bottom')
		#labelStyle = {'color': '#FFF', 'font-size': '12pt'} 
		#ax.setLabel('Set state', units='', **labelStyle)       
		ax.setTicks([[]])   
		self.graph_ann.vb.setMouseEnabled(x=False, y=False)


	def keyPressEvent(self, event):
		#print(event.key())
		# cursor to the right
		if event.key() == 16777236:
			if self.index < self.nbin-5:
				self.index += 1
			self.K[self.index] = 1
			self.plot_eeg()
			self.plot_treck(self.tscale)
			if self.pcollect_index == 1:
				self.index_list.append(self.index)
			else:
				self.index_list = [self.index]
		
		# cursor to the left
		elif event.key() == 16777234:
			if self.index >= 3:
				self.index -= 1
			self.K[self.index] = 1
			self.plot_eeg()
			self.plot_treck(self.tscale)
			if self.pcollect_index == True:
				self.index_list.append(self.index)
			else:
				self.index_list = [self.index]
		
		# r - REM
		elif event.key() == 82:            
			self.M_old = self.M.copy()
			self.M[0,self.index_range()] = 1
			self.index_list = [self.index]
			self.pcollect_index = False
			self.plot_brainstate(self.tscale)
			self.plot_eeg()
		
		# w - Wake
		elif event.key() == 87:
			self.M_old = self.M.copy()
			self.M[0,self.index_range()] = 2
			self.index_list = [self.index]
			self.pcollect_index = False
			self.plot_eeg()
			self.plot_brainstate(self.tscale)
		
		# s or n - SWS/NREM
		elif event.key() == 78 or event.key() == 83:
			self.M_old = self.M.copy()
			self.M[0,self.index_range()] = 3
			self.index_list = [self.index]
			self.pcollect_index = False
			self.plot_eeg()
			self.plot_brainstate(self.tscale)
		# c - Light Green
		elif event.key() == QtCore.Qt.Key_C:
			#self.M[0,self.index] = 0
			self.M_old = self.M.copy()
			self.M[0,self.index_range()] = 4
			self.index_list = [self.index]
			self.pcollect_index = False
			self.plot_eeg()
			self.plot_brainstate(self.tscale)

		# v - RED
		elif event.key() == QtCore.Qt.Key_V:
			#self.M[0,self.index] = 0
			self.M_old = self.M.copy()
			self.M[0,self.index_range()] = 5
			self.index_list = [self.index]
			self.pcollect_index = False
			self.plot_eeg()
			self.plot_brainstate(self.tscale)

		#t - Dark Blue
		elif event.key() == QtCore.Qt.Key_T:
			self.M_old = self.M.copy()
			self.M[0,self.index_range()]=6
			self.index_list = [self.index]
			self.pcollect_index = False
			self.plot_eeg()
			self.plot_brainstate(self.tscale)
		#Yellow
		elif event.key() == QtCore.Qt.Key_B:
			self.M_old = self.M.copy()
			self.M[0,self.index_range()]=7
			self.index_list = [self.index]
			self.pcollect_index = False
			self.plot_eeg()
			self.plot_brainstate(self.tscale)
			
		# z - revert back to previous annotation
		elif event.key() == 90:
			self.M = self.M_old.copy()
			self.plot_eeg()
			self.plot_brainstate(self.tscale)

		# x - undefined state
		elif event.key() == QtCore.Qt.Key_X:
			#self.M[0,self.index] = 0
			self.M_old = self.M.copy()
			self.M[0,self.index_range()] = 0
			self.index_list = [self.index]
			self.pcollect_index = False
			self.plot_eeg()
			self.plot_brainstate(self.tscale)
			
		# space: once space is pressed collect indices starting from space that 
		# are visited with cursor
		elif event.key() == 32:
			self.pcollect_index = True
			self.index_list = [self.index]
		
		# cursor down
		elif event.key() == 16777237:
			self.color_max -= self.color_max/10
			self.image_spectrum.setLevels((0, self.color_max))
		
		# cursor up
		elif event.key() == 16777235:
			self.color_max += self.color_max/10
			self.image_spectrum.setLevels((0, self.color_max))
	   
		# 1 - seconds scale    
		elif event.key() == 49:
			self.tscale = 1.0 
			self.tunit = 's'
			self.plot_session(scale=1, scale_unit='s')
			self.plot_brainstate(scale=1)
			self.plot_treck(scale=1)
			
		# 2 - mintues scale
		elif event.key() == 50:
			self.tscale = 1/60.0 
			self.tunit = 'min'
			self.plot_session(scale=1/60.0, scale_unit='min')
			self.plot_brainstate(scale=1/60.0)
			self.plot_treck(scale=1/60.0)

		# 3 - hours scale
		elif event.key() == 51:
			self.tscale = 1/3600.0 
			self.tunit = 'h'

			self.plot_session(scale=1/3600.0, scale_unit='h')
			self.plot_brainstate(scale=1/3600.0)
			self.plot_treck(scale=1/3600.0)        
		
		# f - save file
		elif event.key() == 70:    
			rewrite_remidx(self.M, self.K, self.ppath, self.name, mode=0)
			#rewrite_stateprob(self.M2, self.K2, self.ppath, self.name)
			self.plot_brainstate(self.tscale)
			self.plot_eeg()
			
		# h - help
		elif event.key() == 72:
			self.print_help()
			
		# e - switch EEG channel
		elif event.key() == 69:
			self.lfp_pointer = -1
			num_eeg = len(self.EEG_list)
			if self.eeg_pointer < num_eeg-1:
				self.eeg_pointer += 1               
			else:
				self.eeg_pointer = 0

			self.EEG = self.EEG_list[self.eeg_pointer]
			self.eeg_spec = self.eeg_spec_list[self.eeg_pointer]
				
			self.plot_eeg()
			#self.plot_treck(self.tscale)
			self.plot_session(scale=self.tscale, scale_unit=self.tunit)                
	
		# m - switch EMG channel
		elif event.key() == 77:
			num_emg = len(self.EMG_list)
			if self.emg_pointer < num_emg-1:
				self.emg_pointer += 1               
			else:
				self.emg_pointer = 0

			self.EMG = self.EMG_list[self.emg_pointer]
			self.EMGAmpl = self.EMGAmpl_list[self.emg_pointer]

			self.plot_eeg()
			self.plot_session(scale=self.tscale, scale_unit=self.tunit)                
		
		# p - switch on/off laser [p]ulses
		elif event.key() == 80:
			if self.pplot_laser==True:
				self.pplot_laser = False
			else:
				self.pplot_laser = True
			self.plot_eeg()
			self.plot_treck(self.tscale)
						
		# l - turn on lfp channel
		elif event.key() == 76:
			self.eeg_pointer = -1
			if len(self.LFP_list) > 0:                                
				num_lfp = len(self.LFP_list)
				if self.lfp_pointer < num_lfp-1:
					self.lfp_pointer += 1
				else:
					self.lfp_pointer = 0                    
				self.EEG = self.LFP_list[self.lfp_pointer]
				self.plot_eeg()

		elif event.key() == QtCore.Qt.Key_I:
			self.print_info()

		# $
		elif event.key() == QtCore.Qt.Key_Dollar:
			self.break_index[1] = len(self.K)-1
			if not self.pbreak:
				self.pbreak = True
			else:
				self.K[self.break_index[0]:self.break_index[1]+1] = -1
				self.pbreak = False
				self.plot_treck(scale=self.tscale)

		# ^
		elif event.key() == 94:
			self.break_index[0] = 0
			if not self.pbreak:
				self.pbreak = True
			else:
				self.K[self.break_index[0]:self.break_index[1]+1] = -1
				self.pbreak = False
				self.plot_treck(scale=self.tscale)

		# [ open break
		elif event.key() == 91:
			self.break_index[0] = int(self.index)
			if not self.pbreak:
				self.pbreak = True
			else:
				self.K[self.break_index[0]:self.break_index[1]+1] = -1
				self.pbreak = False
				self.plot_treck(scale=self.tscale)

		# ]
		elif event.key() == 93:
			self.break_index[1] = int(self.index)
			if not self.pbreak:
				self.pbreak = True
			else:
				self.K[self.break_index[0]:self.break_index[1]+1] = -1
				self.pbreak = False
				self.plot_treck(scale=self.tscale)

		# *
		elif event.key() == 42:
			use_idx = np.where(self.K>=0)[0]
			print("Re-calculating sleep annotation")
			sleepy.sleep_state(ppath, name, th_delta_std=1, mu_std=0, sf=1, sf_delta=3, pwrite=1,
							   pplot=True, pemg=1, vmax=2.5, use_idx=use_idx)
			# reload sleep state
			K_old = self.K.copy()
			(A,self.K) = load_stateidx(self.ppath, self.name)

			# set undefined states to 4
			#A[np.where(A==0)] = 4
			# needs to be packed into 1 x nbin matrix for display
			self.M = np.zeros((1,self.nbin))
			self.M[0,:] = A
			# backup for brainstate in case somethin goes wrong
			self.M_old = self.M.copy()
			self.K[np.where(K_old<0)] = -1
			self.plot_treck(scale=self.tscale)
			self.plot_session(scale=self.tscale, scale_unit=self.tunit)

		event.accept()
		
		
	def index_range(self):
		if len(self.index_list) == 1:
			return self.index_list        
		a = self.index_list[0]
		b = self.index_list[-1]        
		if a<=b:
			return list(range(a,b+1))
		else:
			return list(range(b,a+1))
		

	def mousePressEvent(self, QMouseEvent):
		pos = QMouseEvent.pos()
		# mouse left double-click on Spectrogram, EMG, or treck, or brainstate:
		# jump to the clicked point
		if QMouseEvent.type() == QtCore.QEvent.MouseButtonDblClick:
		
			if self.graph_spectrum.sceneBoundingRect().contains(pos) \
			or self.graph_brainstate.sceneBoundingRect().contains(pos) \
			or self.graph_brainprob.sceneBoundingRect().contains(pos) \
			or self.graph_treck.sceneBoundingRect().contains(pos) or self.graph_emgampl.sceneBoundingRect().contains(pos):
				mousePoint = self.graph_spectrum.vb.mapSceneToView(pos)
				
				self.index = int(mousePoint.x()/(self.fdt*self.tscale))
				#self.index_list = [self.index]
				
				if self.pcollect_index == True:
					self.index_list.append(self.index)
				else:
					self.index_list = [self.index]
								
				#self.pcollect_index = True
				self.plot_eeg()  
				self.plot_treck(self.tscale)
			
		
	def closeEvent(self, event):
		print("Closing...")
		rewrite_remidx(self.M, self.K, self.ppath, self.name, mode=0)
		rewrite_stateprob(self.M2, self.K2, self.ppath, self.name)
		
		
	def openFileNameDialog(self):    
		fileDialog = QFileDialog(self)
		fileDialog.setOption(QFileDialog.ShowDirsOnly, True)
		name = fileDialog.getExistingDirectory(self, "Choose Recording Directory")
		(self.ppath, self.name) = os.path.split(name)        
		print("Setting base folder %s and recording %s" % (self.ppath, self.name))
	 
		
	def load_recording(self):
		"""
		load recording: spectrograms, EEG, EMG, time information etc.
		"""
		if self.name == '':
			self.openFileNameDialog()
		# set title for window
		self.setWindowTitle(self.name)
		
		# load EEG/EMG
		self.eeg_pointer = 0
		self.emg_pointer = 0
		self.EEG_list = []  
		self.EMG_list = []
		self.EMGAmpl_list = []
		
		self.eeg_spec_list = []              
		#self.eeg_amp_list = []
		
		# load EEG1 and EMG1
		EEG1 = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EEG.mat'))['EEG']).astype(np.float32)
		self.EEG_list.append(EEG1)
		EMG1 = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EMG.mat'))['EMG']).astype(np.float32)
		self.EMG_list.append(EMG1)
		# if existing, also load EEG2 and EMG2
		if os.path.isfile(os.path.join(self.ppath, self.name, 'EEG2.mat')):
			EEG2 = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EEG2.mat'))['EEG2']).astype(np.float32)
			self.EEG_list.append(EEG2)

		# and the same for EMG2
		if os.path.isfile(os.path.join(self.ppath, self.name, 'EMG2.mat')):
			EMG2 = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EMG2.mat'))['EMG2']).astype(np.float32)
			self.EMG_list.append(EMG2)

		self.EEG = self.EEG_list[0]
		self.EMG = self.EMG_list[0]
		
		# median of EEG signal to scale the laser signal
		self.eeg_amp = np.median(np.abs(self.EEG))
					   
		# load spectrogram / EMG amplitude
		if not(os.path.isfile(os.path.join(self.ppath, self.name, 'sp_' + self.name + '.mat'))):
			# spectrogram does not exist, generate it
			sleepy.calculate_spectrum(self.ppath, self.name, fres=0.5)
			print("Calculating spectrogram for recording %s\n" % self.name)
		
		spec = so.loadmat(os.path.join(self.ppath, self.name, 'sp_' + self.name + '.mat'))
		self.eeg_spec_list.append(spec['SP'])
		if 'SP2' in spec:
			self.eeg_spec_list.append(spec['SP2'])
		#else:
		#    self.eeg_spec_list.append(spec['SP'])
		self.eeg_spec = self.eeg_spec_list[0]
				
		self.ftime = spec['t'][0]
		self.fdt = spec['dt'][0][0]
		freq = np.squeeze(spec['freq'])
		self.ifreq = np.where(freq <= 25)[0][-1]
		self.fdx = freq[1]-freq[0]
		self.mfreq = np.where((freq>=10) & (freq <= 500))[0]
		self.freq = freq #correct
		
		self.emg_spec = so.loadmat(os.path.join(self.ppath, self.name, 'msp_' + self.name + '.mat'))
		EMGAmpl1 = np.sqrt(self.emg_spec['mSP'][self.mfreq,:].sum(axis=0))
		self.EMGAmpl_list.append(EMGAmpl1)
		if 'mSP2' in self.emg_spec:
			EMGAmpl2 = np.sqrt(self.emg_spec['mSP2'][self.mfreq,:].sum(axis=0))
			self.EMGAmpl_list.append(EMGAmpl2)
		#else:
		#    self.EMGAmpl_list.append(EMGAmpl1)
		self.EMGAmpl = self.EMGAmpl_list[0]
		
		# load LFP signals
		# get all LFP files
		self.lfp_pointer = -1
		self.LFP_list = []
		lfp_files = [f for f in os.listdir(os.path.join(self.ppath, self.name)) if re.match('^LFP', f)]
		lfp_files.sort()
		if len(lfp_files) > 0:
			self.lfp_pointer = -1
			for f in lfp_files:
				#pdb.set_trace()
				key = re.split('\\.', f)[0]
				LFP = so.loadmat(os.path.join(self.ppath, self.name, f), squeeze_me=True)[key]
				self.LFP_list.append(LFP)
			#self.LFP_list.append((self.LFP_list[0]-self.LFP_list[1]))

		# set time bins, sampling rates etc.
		self.nbin = len(self.ftime) #number of bins in fourier time
		self.SR = get_snr(self.ppath, self.name)
		self.dt = 1/self.SR
		self.fbin = np.round((1/self.dt) * self.fdt) # number of sampled point for one fourier bin
		if self.fbin % 2 == 1:
			self.fbin += 1

		#load brainprob
		#(A2,self.K2) = load_stateprob(self.ppath, self.name)
		#self.M2 = np.zeros((1,self.nbin))
		#self.M2[0,:] = A2

		# load brain state
		if not(os.path.isfile(os.path.join(self.ppath, self.name, 'remidx_' + self.name + '.txt'))):
			# predict brain state
			M,S = sleepy.sleep_state(self.ppath, self.name, pwrite=1, pplot=0)
		(A,self.K) = load_stateidx(self.ppath, self.name)
		# set undefined states to 4
		#A[np.where(A==0)] = 4
		# needs to be packed into 1 x nbin matrix for display
		self.M = np.zeros((1,self.nbin))
		self.M[0,:] = A
		# backup for brainstate in case somethin goes wrong
		self.M_old = self.M.copy()
				
		# load laser
		# laser signal in brainstate time
		self.laser = np.zeros((self.nbin,))
		# plot laser?
		self.pplot_laser = False
		# supplementary treck signal; for exampal trigger signal from REM-online detection
		self.suppl_treck = []
		self.psuppl = False
		if os.path.isfile(os.path.join(self.ppath, self.name, 'laser_' + self.name + '.mat')):
			lsr = load_laser(self.ppath, self.name)
			(start_idx, end_idx) = laser_start_end(lsr)
			# laser signal in EEG time
			self.laser_raw = lsr
			self.pplot_laser = True

			if len(start_idx) > 0:
				for (i,j) in zip(start_idx, end_idx) :
					i = int(np.round(i/self.fbin))
					j = int(np.round(j/self.fbin))
					self.laser[i:j+1] = 1
			# recording with REM online: ####
			if os.path.isfile(os.path.join(self.ppath, self.name, 'rem_trig_' + self.name + '.mat')):
				self.psuppl = True
				self.suppl_treck = np.zeros((self.nbin,))
				trig = load_trigger(self.ppath, self.name)
				(start_idx, end_idx) = laser_start_end(trig)
				if len(start_idx) > 0:
					for (i, j) in zip(start_idx, end_idx):
						i = int(np.round(i / self.fbin))
						j = int(np.round(j / self.fbin))
						self.suppl_treck[i:j + 1] = 1
			##################################
		else:
			self.laser_raw = np.zeros((len(self.EEG),), dtype='int8')
			
		# load information of light/dark cycles
		self.dark_cycle = get_cycles(self.ppath, self.name)['dark']
				
		# max color for spectrogram
		self.color_max = np.max(self.eeg_spec)
		

				
	def print_help(self):        
		print("""
		usage
		
		to start run
		"python sleep_annotation_qt.py"
		or
		"python sleep_annotation_qt.py /folder/to/your/recording"
		
		key actions:
		h     -     print help
		r     -     REM
		w     -     Wake
		s|n   -     NREM
		x     -     set undefined state
		space -     set mark, next when you press r,w,s,or n all states between
					space and the current time point will be set to the selected
					state
		z     -     undo last annotation
		f     -     save sleep annotation
		1,2,3 -     second, minute, hour as time unit
		e     -     switch EEG channel
		m     -     switch EMG channel
		p     -     switch laser signal on/off
		l     -     show and switch through LFP channels
		i     -     print content of info.txt file
		[     -     open break (piece of recording not considered for brain state
					classification
		]     -     close break
		^     -     recording break starts at beginning
		$     -     recording break goes till end
		*     -     redo automatic brain state classification
		cursor left, right - go to left, right
		cursor up,down     - change color range of EEG spectrogram
		
		mouse actions: 
		scroll on spectrogram: zoom in / out
		scroll on EMG Ampl.: Change Amplitude of EMG
		scroll on EEG,EMG: change amplitude
		double-click on spectrum: jump to clicked time point
		""")

	def print_info(self):
		fid = open(os.path.join(self.ppath, self.name, 'info.txt'), 'rU')
		lines = fid.readlines()
		fid.close()

		print("")
		print("Content of info.txt file of recording %s." % self.name)
		for line in lines:
			if not re.match('^\s+$', line):
				print(line)




# some input parameter management
params = sys.argv[1:]
if (len(params) == 0) :
	ppath = ''
	name = ''
elif len(params) == 1:
	if re.match('.*\/$', params[0]):
		params[0] = params[0][:-1]
	(ppath, name) = os.path.split(params[0])      
else:
	ppath = params[0]
	name  = params[1]

app = QtWidgets.QApplication([])
w = MainWindow(ppath, name)
w.show()
app.exec_()