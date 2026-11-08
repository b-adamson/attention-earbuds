# Data

Keep datasets outside the repo, e.g. in a `B3-Data` folder next to it. Don't commit data.

## MAD-EEG

Cantisani et al. (2019), SMM19, doi:10.21437/SMM.2019-11. https://zenodo.org/record/4537751 (CC BY-SA 4.0)

20-channel EEG at 256 Hz, listeners attending to one instrument in a duo/trio. The preprocessed file (`madeeg_preprocessed.hdf5` + `.yaml`, 3.7 GB) has 6 subjects, 246 trials, with EEG aligned to the mix and each instrument's isolated track.

Electrodes are frontal/central/parietal/occipital only (no temporal or ear sites).
