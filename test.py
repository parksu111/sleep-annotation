import sys
# exactly this combination of imports works with py2.7; otherwise problem importing
# pyqtgraph
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import pyqtgraph as pg
import numpy as np
import scipy.io as so
import os
import re
import h5py
import sleepy
import pdb

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

		# draw in all data for the first time
		self.show()
		self.load_recording()
		self.gen_layout()
		self.plot_high_spectrum()
		self.plot_spectrum()
		self.plot_emgampl()
		self.plot_brainstate()
		self.plot_eeg()
		self.plot_treck()
		
		# initial setup for live EEG, EMG
		self.graph_eeg.setRange(yRange=(-500, 500),  padding=None)
		self.graph_emg.setRange(yRange=(-500, 500),  padding=None)

	def gen_layout(self):
		# General layout
		self.view = pg.GraphicsView()
		self.lay = pg.GraphicsLayout()

		# Brainstate (includes treck, hypnogram, spectrogram)
		self.lay_brainstate = self.lay.addLayout()

		## Treck: What is annotated, laswer, current time point
		self.graph_treck = pg.PlotItem()
		self.lay_brainstate.addItem(self.graph_treck)

		## Hypnogram (color-coded brainstate)
		self.lay_brainstate.nextRow()
		self.graph_brainstate = self.lay_brainstate.addPlot()
		self.image_brainstate = pg.ImageItem()
		self.graph_brainstate.addItem(self.image_brainstate)
		
		## High frequency Spectrogram
		self.lay_brainstate.nextRow()
		self.graph_high_spectrum = self.lay_brainstate.addPlot()
		self.image_high_spectrum = pg.ImageItem()
		self.graph_high_spectrum.addItem(self.image_high_spectrum)

		## Spectrogram
		self.lay_brainstate.nextRow()
		self.graph_spectrum = self.lay_brainstate.addPlot()
		self.image_spectrum = pg.ImageItem()
		self.graph_spectrum.addItem(self.image_spectrum)

		## EMG Amplitude
		self.lay_brainstate.nextRow()
		self.graph_emgampl = self.lay_brainstate.addPlot()

		# Raw Data
		self.lay.nextRow()

		## EEG
		self.graph_eeg = pg.PlotItem()
		self.lay.addItem(self.graph_eeg)

		## EMG
		self.lay.nextRow()
		self.graph_emg = pg.PlotItem()
		self.lay.addItem(self.graph_emg)

		# Annotation Graph
		self.lay.nextRow()
		self.lay_ann = self.lay.addLayout()
		self.graph_ann = pg.PlotItem()
		self.lay_ann.addItem(self.graph_ann)

		# Organize windows
		self.view.setCentralItem(self.lay)
		self.setCentralWidget(self.view)

		# Colormaps
		## Hypnogram
		pos = np.linspace(0, 1, 8)
		color = np.array(
			[[0, 0, 0, 200], [0, 255, 255, 200], [150, 0, 255, 200], [150, 150, 150, 200], [0,255,43,200], [255,20,20,200], [66,86,219,200], [255,255,0,200]], dtype=np.ubyte)
		cmap = pg.ColorMap(pos, color)
		self.lut_brainstate = cmap.getLookupTable(0.0, 1.0, 8)

		## Spectrogram
		pos = np.array([0., 0.05, .2, .4, .6, .9])
		color = np.array([[0, 0, 0, 255], [0,0,128,255], [0,255,0,255], [255,255,0, 255], (255,165,0,255), (255,0,0, 255)], dtype=np.ubyte)
		cmap = pg.ColorMap(pos, color)
		self.lut_spectrum = cmap.getLookupTable(0.0, 1.0, 256)

	def plot_high_spectrum(self, scale=1):
		self.graph_high_spectrum.clear()
		self.image_high_spectrum = pg.ImageItem()
		self.graph_high_spectrum.addItem(self.image_high_spectrum)

		self.image_high_spectrum.setImage(self.eeg_highspec[self.high_freq,:].T)

		# Scale image to seconds/minutes/hours
		tr = QtGui.QTransform()
		tr.scale(self.fdt*scale, 1.0*self.fdx)
		self.image_high_spectrum.setTransform(tr)

		# Allow mouse-scroll along x axis
		self.graph_high_spectrum.vb.setMouseEnabled(x=True, y=False)

		# Limits
		limits = {'xMin': -1*self.fdt*scale, 'xMax': self.ftime[-1]*scale, 'yMin': 0, 'yMax': self.freq[self.high_freq[-1]]}
		self.graph_high_spectrum.vb.setLimits(**limits)

		# Link graph with self.graph_brainstate
		self.graph_high_spectrum.setXLink(self.graph_brainstate.vb)

		# Label y-axis
		ax = self.graph_high_spectrum.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('Freq', units='Hz', **labelStyle)
		ax.setTicks([[(0, '300'), (100, '400'), (200, '500')]])

		# Set colormap
		self.image_high_spectrum.setLookupTable(self.lut_spectrum)

	def plot_spectrum(self, scale=1, scale_unit='s'):
		# Clear plot and reload ImageItem
		self.graph_spectrum.clear()
		self.image_spectrum = pg.ImageItem()
		self.graph_spectrum.addItem(self.image_spectrum)

		# Set image
		self.image_spectrum.setImage(self.eeg_spec[0:self.ifreq,:].T)

		# Scale image to seconds/minutes/hours
		tr = QtGui.QTransform()
		tr.scale(self.fdt*scale, 1.0*self.fdx)
		self.image_spectrum.setTransform(tr)

		# Allow mouse-scroll along x axis
		self.graph_spectrum.vb.setMouseEnabled(x=True, y=False)

		# Limits
		limits = {'xMin': -1*self.fdt*scale, 'xMax': self.ftime[-1]*scale, 'yMin': 0, 'yMax': self.freq[self.ifreq]}
		self.graph_spectrum.vb.setLimits(**limits)

		# Link graph with self.graph_brainstate
		self.graph_spectrum.setXLink(self.graph_brainstate.vb)

		# Label y-axis
		ax = self.graph_spectrum.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('Freq', units='Hz', **labelStyle)
		ax.setTicks([[(0, '0'), (10, '10'), (20, '20')]])

		# Set colormap
		self.image_spectrum.setLookupTable(self.lut_spectrum)

	def plot_emgampl(self, scale=1, scale_unit='s'):
		# Clear plot
		self.graph_emgampl.clear()
		
		# Plot
		self.graph_emgampl.plot(self.ftime*scale + scale*self.fdt/2.0, self.EMGAmpl)

		# Allow mouse-scroll along y-axis
		self.graph_emgampl.vb.setMouseEnabled(x=False, y=True)

		# Limits
		limits = {'xMin': 0, 'xMax': self.ftime[-1]*scale}
		self.graph_emgampl.vb.setLimits(**limits)		

		# Link graph
		self.graph_emgampl.setXLink(self.graph_spectrum.vb)

		# Label y-axis
		ax = self.graph_emgampl.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('EMG Ampl.', units='V', **labelStyle)

		# Label x-axis
		ax = self.graph_emgampl.getAxis(name='bottom')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('Time', units=scale_unit, **labelStyle)

	def plot_brainstate(self, scale=1):
		# Clear plot and reload ImageItem
		self.graph_brainstate.clear()
		self.image_brainstate = pg.ImageItem() 
		self.graph_brainstate.addItem(self.image_brainstate)

		# Set image
		self.image_brainstate.setImage(self.M.T)

		# Scale
		tr = QtGui.QTransform()
		tr.scale(self.fdt*scale,1)
		self.image_brainstate.setTransform(tr)

		# Allow mouse-scroll along x-axis
		self.graph_brainstate.vb.setMouseEnabled(x=True, y=False)

		# Link graph
		self.graph_brainstate.setXLink(self.graph_spectrum.vb)

		# Limits
		limits = {'xMin': -1*self.fdt*scale, 'xMax': self.ftime[-1]*scale, 'yMin': 0, 'yMax': 1}
		self.graph_brainstate.vb.setLimits(**limits)

		# Label y-axis
		ax = self.graph_brainstate.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('Brainstate', units='', **labelStyle)
		ax.setTicks([[(0, ''), (1, '')]])

		# Label x-axis
		ax = self.graph_brainstate.getAxis(name='bottom')
		ax.setTicks([[]])

		# Set colormap
		self.image_brainstate.setLookupTable(self.lut_brainstate)
		self.image_brainstate.setLevels([0, 7])

	def plot_treck(self, scale=1):
		# Clear graph
		self.graph_treck.clear()

		# Plot
		self.graph_treck.plot(self.ftime*scale, self.K[0:self.nbin]*0.5, pen=(150,150,150))
		## Plot dark cycles
		self.graph_treck.plot(self.ftime, np.zeros((self.ftime.shape[0],)), pen=pg.mkPen(width=8, color='w'))
		for d in self.dark_cycle:
			a = int(d[0]/self.fdt)
			b = int(d[1]/self.fdt)
			self.graph_treck.plot(self.ftime[a:b+1]*scale, np.zeros((b-a+1,)), pen=pg.mkPen(width=8, color=(100, 100, 100)))
		## plot laser
		if self.pplot_laser:
			self.graph_treck.plot(self.ftime*scale, self.laser, pen=(0,0,255))
		## Plot supplemental signal
		if self.psuppl:
			self.graph_treck.plot(self.ftime*scale, self.suppl_treck*0.3, pen=(255,150,150))
		## Plot currently annotated point
		self.graph_treck.plot([self.ftime[self.index]*scale + 0.5*self.fdt*scale], [0.0], pen=(0, 0, 0), symbolPen=(255, 0, 0), symbolBrush=(255, 0, 0), symbolSize=5)		

		# Limits
		limits = {'xMin': -1*self.fdt*scale, 'xMax': self.ftime[-1]*scale, 'yMin': -1.1, 'yMax': 1.1}
		self.graph_treck.vb.setLimits(**limits)

		# Link graph
		self.graph_treck.setXLink(self.graph_spectrum.vb)

		# Label y-axis
		ax = self.graph_treck.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('Laser', units='', **labelStyle)
		ax.setTicks([[(0, ''), (1, '')]])

		# Label x-axis: remove ticks on x-axis
		ax = self.graph_treck.getAxis(name='bottom')
		ax.setTicks([[]])

	def plot_eeg(self):
		timepoint = self.ftime[self.index] + self.fdt
		self.twin = 2*self.fdt
		
		n = int(np.round((self.twin+self.fdt/2)/self.dt))
		i = int(np.round(timepoint / self.dt))
		ii = np.arange(i-n, i+n+1)
		t = np.arange(timepoint-n*self.dt, timepoint+n*self.dt+self.dt/2, self.dt)

		# EEG
		## clear graph
		self.graph_eeg.clear()
		## set Range
		self.graph_eeg.setRange(xRange=(t[0],t[-1]), padding=None)
		## Plot EEG
		self.graph_eeg.plot(t,self.EEG[ii]) 
		## Label y-axis
		ax = self.graph_eeg.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('EEG' + ' ' + str(self.eeg_pointer+1), units='V', **labelStyle)
		## Label x-axis
		ax = self.graph_eeg.getAxis(name='bottom')
		ax.setTicks([[]])
		## Set mouse-scroll
		self.graph_eeg.vb.setMouseEnabled(x=False, y=True)

		# PLOT LASER??????
		#if self.pplot_laser == True:
		#	self.graph_eeg.plot(t, self.laser_raw[ii]*self.eeg_amp*2, pen=(0,0,255))

		# EMG
		## clear graph
		self.graph_emg.clear()
		## set Range
		self.graph_emg.setRange(xRange=(t[0],t[-1]), padding=None)
		## plot EMG signal
		self.graph_emg.plot(t,self.EMG[ii])
		## Label y-axis
		ax = self.graph_emg.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('EMG' + ' ' + str(self.emg_pointer+1), units='V', **labelStyle)
		## Label x-axis
		ax = self.graph_emg.getAxis(name='bottom')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'}
		ax.setLabel('Time', units='s', **labelStyle)
		## Set mouse_scroll
		self.graph_emg.vb.setMouseEnabled(x=False, y=True)

		# Sleep stage annotation
		## Indices
		n = int(self.twin/self.fdt);
		i = self.index;
		ii = list(range(i-n,i+n+1))
		## clear graph
		self.graph_ann.clear()
		## set range
		self.graph_ann.setRange(yRange=(1, 3), xRange=(0,5), padding=None)
		## plot
		self.graph_ann.plot(np.arange(0,5)+0.5, self.M[0,ii], name = 'Ann', pen=(255,0,0), symbolPen='w')
		## label y-axis
		ax = self.graph_ann.getAxis(name='left')
		labelStyle = {'color': '#FFF', 'font-size': '12pt'} 
		ax.setLabel('State', units='', **labelStyle)       
		ax.setTicks([[(1, 'R'), (2, 'W'), (3, 'S')]])
		## label x-axis		
		ax = self.graph_ann.getAxis(name='bottom')      
		ax.setTicks([[]])
		## disable mouse-scroll
		self.graph_ann.vb.setMouseEnabled(x=False, y=False)

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
			or self.graph_high_spectrum.sceneBoundingRect().contains(pos) \
			or self.graph_brainstate.sceneBoundingRect().contains(pos) \
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
		sleepy.write_remidx(self.M, self.K, self.ppath, self.name, mode=0)
		#sleepy.rewrite_remidx(self.M, self.K, self.remidx)
		
		
	def openFileNameDialog(self):    
		fileDialog = QFileDialog(self)
		fileDialog.setOption(QFileDialog.ShowDirsOnly, True)
		name = fileDialog.getExistingDirectory(self, "Choose Recording Directory")
		(self.ppath, self.name) = os.path.split(name)        
		print("Setting base folder %s and recording %s" % (self.ppath, self.name))
		self.remidx = os.path.join(self.ppath, self.name, 'remidx_' + self.name + '.txt')

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
		self.eeg_spec = self.eeg_spec_list[0]
				
		self.ftime = spec['t'][0]
		self.fdt = spec['dt'][0][0]
		freq = np.squeeze(spec['freq'])
		self.ifreq = np.where(freq <= 25)[0][-1]
		self.fdx = freq[1]-freq[0]
		self.mfreq = np.where((freq>=10) & (freq <= 500))[0]
		self.freq = freq #correct

		# EEG spectrogram for MA detection
		highspec = spec.copy()
		self.eeg_highspec = highspec['SP']
		self.high_freq = np.where((freq>=300)&(freq<=500))[0]
		
		self.emg_spec = so.loadmat(os.path.join(self.ppath, self.name, 'msp_' + self.name + '.mat'))
		EMGAmpl1 = np.sqrt(self.emg_spec['mSP'][self.mfreq,:].sum(axis=0))
		self.EMGAmpl_list.append(EMGAmpl1)
		if 'mSP2' in self.emg_spec:
			EMGAmpl2 = np.sqrt(self.emg_spec['mSP2'][self.mfreq,:].sum(axis=0))
			self.EMGAmpl_list.append(EMGAmpl2)
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
				key = re.split('\\.', f)[0]
				LFP = so.loadmat(os.path.join(self.ppath, self.name, f), squeeze_me=True)[key]
				self.LFP_list.append(LFP)

		# set time bins, sampling rates etc.
		self.nbin = len(self.ftime) #number of bins in fourier time
		self.SR = sleepy.get_snr(self.ppath, self.name)
		self.dt = 1/self.SR
		self.fbin = np.round((1/self.dt) * self.fdt) # number of sampled point for one fourier bin
		if self.fbin % 2 == 1:
			self.fbin += 1

		# load brain state
		#if not(os.path.isfile(os.path.join(self.ppath, self.name, 'remidx_' + self.name + '.txt'))):
		if not(os.path.isfile(os.path.join(self.ppath, self.name, 'remidx_' + self.name + '.txt'))):
			# predict brain state
			M,S = sleepy.sleep_state(self.ppath, self.name, pwrite=1, pplot=0)
		(A,self.K) = sleepy.load_stateidx(self.ppath, self.name)
		# set undefined states to 4
		#A[np.where(A==0)] = 4
		# needs to be packed into 1 x nbin matrix for display
		self.M = np.zeros((1,self.nbin))
		self.M[0,:] = A[0:self.nbin]
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
			lsr = sleepy.load_laser(self.ppath, self.name)
			(start_idx, end_idx) = sleepy.laser_start_end(lsr)
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
				trig = sleepy.load_trigger(self.ppath, self.name)
				(start_idx, end_idx) = sleepy.laser_start_end(trig)
				if len(start_idx) > 0:
					for (i, j) in zip(start_idx, end_idx):
						i = int(np.round(i / self.fbin))
						j = int(np.round(j / self.fbin))
						self.suppl_treck[i:j + 1] = 1
			# REM deprivation   
		elif os.path.isfile(os.path.join(self.ppath, self.name, 'pull_' + self.name + '.mat')):
			self.psuppl = True
			self.suppl_treck = np.zeros((self.nbin,))
			trig = load_pull(self.ppath, self.name)
			(start_idx, end_idx) = sleepy.laser_start_end(trig)
			if len(start_idx) > 0:
				for (i, j) in zip(start_idx, end_idx):
					i = int(np.round(i / self.fbin))
					j = int(np.round(j / self.fbin))
					self.suppl_treck[i:j + 1] = 1                                    
			##################################
		else:
			self.laser_raw = np.zeros((len(self.EEG),), dtype='int8')
			
		# load information of light/dark cycles
		#self.dark_cycle = sleepy.get_cycles(self.ppath, self.name)['dark']
		self.dark_cycle = sleepy.find_dark(self.ppath, self.name)
				
		# max color for spectrogram
		self.color_max = np.max(self.eeg_spec)

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
			#self.plot_session(scale=1, scale_unit='s')
			self.plot_spectrum(scale=1)
			self.plot_emgampl(scale=1, scale_unit='s')			
			self.plot_brainstate(scale=1)
			self.plot_treck(scale=1)
			
		# 2 - mintues scale
		elif event.key() == 50:
			self.tscale = 1/60.0 
			self.tunit = 'min'
			#self.plot_session(scale=1/60.0, scale_unit='min')
			self.plot_spectrum(scale=1/60.0)
			self.plot_emgampl(scale=1/60.0, scale_unit='min')
			self.plot_brainstate(scale=1/60.0)
			self.plot_treck(scale=1/60.0)

		# 3 - hours scale
		elif event.key() == 51:
			self.tscale = 1/3600.0 
			self.tunit = 'h'

			#self.plot_session(scale=1/3600.0, scale_unit='h')
			self.plot_spectrum(scale=1/3600.0)
			self.plot_emgampl(scale=1/3600.0, scale_unit='h')
			self.plot_brainstate(scale=1/3600.0)
			self.plot_treck(scale=1/3600.0)        
		
		# f - save file
		elif event.key() == 70:    
			#rewrite_remidx(self.M, self.K, self.ppath, self.name, mode=0)
			rewrite_remidx(self.M, self.K, self.remidx)
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
			self.plot_treck(self.tscale)
			#self.plot_session(scale=self.tscale, scale_unit=self.tunit)
			self.plot_spectrum(scale=self.tscale)
			self.plot_emgampl(scale=self.tscale, scale_unit=self.tunit)
	
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
			#self.plot_session(scale=self.tscale, scale_unit=self.tunit)        
			self.plot_spectrum(scale=self.tscale)
			self.plot_emgampl(scale=self.tscale, scale_unit=self.tunit)
		
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
			sleepy.sleep_state(self.ppath, self.name, th_delta_std=1, mu_std=0, sf=1, sf_delta=3, pwrite=1,
							   pplot=True, pemg=1, vmax=2.5, use_idx=use_idx)
			# reload sleep state
			K_old = self.K.copy()
			(A,self.K) = sleepy.load_stateidx(self.ppath, self.name, self.remidx)

			# set undefined states to 4
			#A[np.where(A==0)] = 4
			# needs to be packed into 1 x nbin matrix for display
			self.M = np.zeros((1,self.nbin))
			self.M[0,:] = A[0:self.nbin]
			# backup for brainstate in case somethin goes wrong
			self.M_old = self.M.copy()
			self.K[np.where(K_old<0)] = -1
			self.plot_treck(scale=self.tscale)
			#self.plot_session(scale=self.tscale, scale_unit=self.tunit)
			self.plot_spectrum(scale=self.tscale)
			self.plot_emgampl(scale=self.tscale, scale_unit=self.tunit)

		event.accept()		

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