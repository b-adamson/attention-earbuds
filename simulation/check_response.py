"""Is there a music response in MAD-EEG, and is it bigger for the attended instrument?

For each subject (duo trials only):
  - fit a TRF from each instrument's loudness to the EEG, leaving one song out at a time
  - score how well it predicts the held-out EEG (correlation)
  - compare attended vs ignored instrument, and a time-shifted control (no real link)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from load_madeeg import list_trials, load_meta, load_trial

FS = 64
TMAX = 0.5
ALPHA = 1.0
OUT = Path(__file__).resolve().parent / "results"


def prep_eeg(eeg, fs):
    sos = signal.butter(4, [1, 8], btype="band", fs=fs, output="sos")
    eeg = signal.sosfiltfilt(sos, eeg, axis=1)
    eeg = signal.resample(eeg, int(eeg.shape[1] * FS / fs), axis=1)
    return zscore(eeg)


def envelope(audio, fs, n):
    env = np.abs(signal.hilbert(audio))
    sos = signal.butter(4, 8, btype="low", fs=fs, output="sos")
    return zscore(signal.resample(signal.sosfiltfilt(sos, env), n))


def zscore(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-12)


def lagged(x, n_lags):
    X = np.zeros((len(x), n_lags))
    for i in range(n_lags):
        X[i:, i] = x[: len(x) - i]
    return X


def fit(xs, ys, n_lags):
    X = np.vstack([lagged(x, n_lags) for x in xs])
    Y = np.hstack(ys).T
    A = X.T @ X
    return np.linalg.solve(A + ALPHA * np.trace(A) / n_lags * np.eye(n_lags), X.T @ Y)


def score(x, y, w, n_lags):
    p = (lagged(x, n_lags) @ w).T
    p, y = zscore(p), zscore(y)
    return float(np.mean(np.mean(p * y, axis=1)))


def load_subject(meta, subject):
    trials = []
    for s, name in list_trials(meta):
        if s != subject or "_duo_" not in name:
            continue
        t = load_trial(meta, s, name)
        eeg = prep_eeg(t["eeg"], t["eeg_fs"])
        envs = [envelope(st, t["audio_fs"], eeg.shape[1]) for st in t["stems"]]
        a = t["target"]
        trials.append({"song": meta[s][name]["song"], "eeg": eeg,
                       "att": envs[a], "ign": envs[1 - a], "channels": t["channels"]})
    return trials


def evaluate(trials, n_lags):
    shift = 5 * FS
    res = {"attended": [], "ignored": [], "shifted": []}
    for song in sorted({t["song"] for t in trials}):
        train = [t for t in trials if t["song"] != song]
        test = [t for t in trials if t["song"] == song]
        if not train or not test:
            continue
        w_att = fit([t["att"] for t in train], [t["eeg"] for t in train], n_lags)
        w_ign = fit([t["ign"] for t in train], [t["eeg"] for t in train], n_lags)
        for t in test:
            res["attended"].append(score(t["att"], t["eeg"], w_att, n_lags))
            res["ignored"].append(score(t["ign"], t["eeg"], w_ign, n_lags))
            res["shifted"].append(score(np.roll(t["att"], shift), t["eeg"], w_att, n_lags))
    w_all = fit([t["att"] for t in trials], [t["eeg"] for t in trials], n_lags)
    return {k: np.mean(v) for k, v in res.items()}, w_all


def main():
    meta = load_meta()
    subjects = sorted(meta)
    n_lags = int(TMAX * FS) + 1
    OUT.mkdir(exist_ok=True)

    rows, trfs, channels = [], [], None
    for s in subjects:
        cache = OUT / f"cache_{s}.npz"
        if cache.exists():
            c = np.load(cache, allow_pickle=True)
            r, w, channels = c["r"].item(), c["w"], list(c["channels"])
        else:
            trials = load_subject(meta, s)
            if len(trials) < 2:
                continue
            channels = trials[0]["channels"]
            r, w = evaluate(trials, n_lags)
            np.savez(cache, r=r, w=w, channels=channels)
        rows.append((s, 0, r["attended"], r["ignored"], r["shifted"]))
        trfs.append(w)
        print(f"{s}: attended r={r['attended']:.4f} "
              f"ignored r={r['ignored']:.4f} shifted r={r['shifted']:.4f}")

    att, ign, sh = (np.array([r[i] for r in rows]) for i in (2, 3, 4))
    print(f"\nmean   attended {att.mean():.4f}  ignored {ign.mean():.4f}  shifted {sh.mean():.4f}")
    print(f"attended > shifted in {np.sum(att > sh)}/{len(rows)} subjects")
    print(f"attended > ignored in {np.sum(att > ign)}/{len(rows)} subjects")

    lags_ms = np.arange(n_lags) / FS * 1000
    trf = np.mean(trfs, axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for ch in ["Fz", "Cz", "Pz"]:
        ax[0].plot(lags_ms, trf[:, channels.index(ch)], label=ch)
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set(xlabel="lag (ms)", ylabel="weight", title="Average TRF, attended instrument")
    ax[0].legend()
    x = np.arange(len(rows))
    ax[1].bar(x - 0.25, att, 0.25, label="attended")
    ax[1].bar(x, ign, 0.25, label="ignored")
    ax[1].bar(x + 0.25, sh, 0.25, label="shifted (control)")
    ax[1].set_xticks(x, [r[0] for r in rows])
    ax[1].set(xlabel="subject", ylabel="prediction r (held-out song)", title="How well loudness predicts EEG")
    ax[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "music_response_check.png", dpi=150)
    print(f"\nsaved {OUT / 'music_response_check.png'}")


if __name__ == "__main__":
    main()
