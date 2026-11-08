from pathlib import Path
import h5py
import numpy as np
import yaml

DATA = Path(__file__).resolve().parents[2] / "B3-Data" / "madeeg"


def load_meta():
    with open(DATA / "madeeg_preprocessed.yaml") as f:
        return yaml.safe_load(f)


def list_trials(meta):
    return [(s, t) for s in meta for t in meta[s]]


def load_trial(meta, subject, name):
    m = meta[subject][name]
    n = len(m["instruments"])
    with h5py.File(DATA / "madeeg_preprocessed.hdf5", "r") as f:
        g = f[subject][name]
        eeg = np.array(g["response"])
        stems = np.array(g["soli"])[:n]
    return {
        "eeg": eeg,
        "eeg_fs": m["eeg_info"]["sfreq"],
        "channels": m["eeg_info"]["ch_names"],
        "stems": stems,
        "audio_fs": m["wav_info"]["sfreq"],
        "instruments": m["instruments"],
        "target": m["instruments"].index(m["target"]),
    }
