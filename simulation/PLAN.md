# Simulator plan

Goal: generate fake EEG of someone listening to a song while focusing on one instrument.

## Steps
1. Load MAD-EEG (real EEG + each instrument's track).
2. Get each instrument's loudness over time.
3. Learn how the brain responds to loudness (TRF) from real EEG.
4. Generate: response to each instrument, attended one boosted, plus background noise.
5. Compare fake vs real.
6. Later: real background EEG, artefacts, headset electrode layouts.

## MT5
- [ ] Load one trial, plot EEG next to instrument loudness
- [ ] Fit a TRF; check there's a clear response
- [ ] Generate a first fake trial and compare to real
- [ ] Agree data format with Cody
