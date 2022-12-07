# Sleep Annotation Tool

This tool is an updated version of the sleep annotation tool written by Franz Weber ([github link](https://github.com/tortugar/Lab)) and used at the [Weber Lab](https://www.med.upenn.edu/weberlab/) of the Perelman School of Medicine, University of Pennsylvania.

## Basic Functionality

This program allows users to view raw EEG & EMG signals and annotate each 2.5 second bin according to its state: Wake, NREM, REM.
*add image*

There are 8 rows, each displaying different information that can be used to help label the data:
1. **Laser** - Indicates whether or not a laser was on for optogenetic stimulation of specific brain regions.
2. **Brainstate** - Color coded hypnogram representing the state of the brain.
  * REM - Cyan
  * NREM - Gray
  * WAKE - Purple
  * Intermediate - Dark Blue
  * Undefined - Black
3. Spectogram - Spectogram of the raw EEG signal.
4. EMG Amplitude - Amplitude of the raw EMG signal.
5. EEG - Raw EEG signal.
6. EMG - RAW EMG signal.
7. Zpectogram 2 - Spectogram zoomed into each 2.5 second time bin.
8. State - Chart representing the state of the brain.

## How to Use

### Start
```
python saqt.py $/path/to/folder
```
