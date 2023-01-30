# Sleep Annotation Tool

This tool is an updated version of the sleep annotation tool written by Franz Weber ([github link](https://github.com/tortugar/Lab)) and used at the [Weber Lab](https://www.med.upenn.edu/weberlab/) of the Perelman School of Medicine, University of Pennsylvania.

## Basic Functionality

This program allows users to view raw EEG & EMG signals and annotate each 2.5 second bin according to its state: Wake, NREM, REM.
![alt text](https://github.com/parksu111/sleep-annotation/blob/main/img/outline.png)

There are 9 rows, each displaying different information that can be used to help label the data:
1. **Treck** - Indicates whether or not a laser was on for optogenetic stimulation of specific brain regions.
  * White/Black bar indicates phase (White = light, Black = dark)
2. **Issue** - Indicates whether or not there is an improper state transition .
3. **State** - Color coded hypnogram representing the state of the brain.
  * REM - Cyan
  * NREM - Gray
  * WAKE - Purple
  * Intermediate - Dark Blue
  * Undefined - Black
4. **High Spectrogram** - Spectrogram of raw EEG signal in 300~500Hz range.
5. **Spectrogram** - Spectogram of the raw EEG signal in 0~20Hz range.
6. **EMG Amplitude** - Amplitude of the raw EMG signal.
7. **EEG** - Raw EEG signal.
8. **EMG** - RAW EMG signal.
9. **Brain State** - Chart representing the state of the brain.

## How to Use

### Conda environment
Create a new python environment using the environment.yml file:
```
conda env create -f environment.yml
```
The environment will be named "skynet2". If you want a different name, open the yml file with a text editor and change the name.

### Start
To start the program, run the following command:
```
python sleep_annotation_qt.py $/path/to/folder
```

### Key Actions
#### General
* Left / Right - move to left or right time bin
* Up / Down - Change color range of EEG spectrogram
* Home / End - Change color range of high EEG spectrogram
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
* Scroll on spectrogram / State / Treck: zoom in / out
* Scroll on EMG Ampl.: change Amplitude of EMG
* Scroll on EEG,EMG: change amplitude
* Double-click on spectrum: jump to clicked time point


## Added Functionality
* Add 'black' to **Treck** to indicate dark phase (7pm~7am)
* Add **Issue** bar to indicate bins where improper state transitions occur (red)
* Add **High spectrogram** to help detect microarousals
* Add New Brain States
  * Intermediate (Dark Blue)
  * Microarousal (Green)
* Re-order label in **Brain state** graph: Bottom - W, Middle - N, Top - R
* Add vertical lines to **Brain state** to indicate 2.5s bins

## To-do
- [ ] Add support for edf file format
- [ ] Integrate automatic sleep classification


