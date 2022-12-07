# Sleep Annotation Tool

This tool is an updated version of the sleep annotation tool written by Franz Weber ([github link](https://github.com/tortugar/Lab)) and used at the [Weber Lab](https://www.med.upenn.edu/weberlab/) of the Perelman School of Medicine, University of Pennsylvania.

## Basic Functionality

This program allows users to view raw EEG & EMG signals and annotate each 2.5 second bin according to its state: Wake, NREM, REM.
![alt text](https://github.com/parksu111/sleep-annotation/blob/main/img/gui.png)

There are 8 rows, each displaying different information that can be used to help label the data:
1. **Laser** - Indicates whether or not a laser was on for optogenetic stimulation of specific brain regions.
2. **Brainstate** - Color coded hypnogram representing the state of the brain.
  * REM - Cyan
  * NREM - Gray
  * WAKE - Purple
  * Intermediate - Dark Blue
  * Undefined - Black
3. **Spectogram** - Spectogram of the raw EEG signal.
4. **EMG Amplitude** - Amplitude of the raw EMG signal.
5. **EEG** - Raw EEG signal.
6. **EMG** - RAW EMG signal.
7. **Spectogram 2** - Spectogram zoomed into each 2.5 second time bin.
8. **State** - Chart representing the state of the brain.

## How to Use

### Start
To start the program, run the following command:
```
python saqt.py $/path/to/folder
```

### Key Actions
#### General
* Left / Right - move to left or right time bin
* Up / Down - Change color range of EEG spectogram
* h - print help
* z - undo last annotation
* f - save sleep annotation
* i - print content of info.txt file
#### Channel control
* e - switch between EEG channels
* m - switch between EMG channels
#### Annotating specific time bin
* r - REM
* s / n - NREM
* w - WAKE
* x - Undefined
* t - Intermediate
* c - Microarousal
* space - set mark: the next time you press a key for a specific 
  state (r,w,s,n,t), all time bins between the makr and current time point
  will be set to the selected state
#### Time Unit
* 1 - Seconds
* 2 - Minutes
* 3 - Hours

### Mouse Actions
* Scroll on spectrogram: zoom in / out
* Scroll on EMG Ampl.: change Amplitude of EMG
* Scroll on EEG,EMG: change amplitude
* Double-click on spectrum: jump to clicked time point


## Added Functionality
* Switch between different EEG and EMG channels
* View EMG amplitude to help label 'WAKE' state
* New Brain States
  * Intermediate (Dark Blue)
  * Microarousal (Green)

## To-do
- [ ] Add support for edf file format
- [ ] Integrate automatic sleep classification
- [ ] Add high frequency EMG spectogram
- [ ] Fix state chart to include new states


