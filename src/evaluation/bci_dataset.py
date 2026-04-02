"""BCI Competition IV Dataset 2a loader for downstream evaluation.

4-class motor imagery: left hand, right hand, feet, tongue.
22 EEG channels, 250Hz → resampled to 200Hz, 2-second windows.

Standard protocol:
  - Subject-dependent: session 1 = train, session 2 = test
  - Cross-subject (LOSO): train on 8 subjects, test on 1

Usage:
    train_ds, test_ds, coords = load_bci_subject(subject_id=1)
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# 22 EEG channel names in BCI IV 2a (10-10 system)
BCI_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

# Approximate 3D coordinates for BCI IV 2a channels (unit sphere)
BCI_COORDS = {
    "Fz":  ( 0.00,  0.72,  0.69),
    "FC3": (-0.55,  0.34,  0.77), "FC1": (-0.25,  0.36,  0.90),
    "FCz": ( 0.00,  0.36,  0.93), "FC2": ( 0.25,  0.36,  0.90),
    "FC4": ( 0.55,  0.34,  0.77),
    "C5":  (-0.95,  0.00,  0.31), "C3":  (-0.71,  0.00,  0.71),
    "C1":  (-0.33,  0.00,  0.94), "Cz":  ( 0.00,  0.00,  1.00),
    "C2":  ( 0.33,  0.00,  0.94), "C4":  ( 0.71,  0.00,  0.71),
    "C6":  ( 0.95,  0.00,  0.31),
    "CP3": (-0.55, -0.34,  0.77), "CP1": (-0.25, -0.36,  0.90),
    "CPz": ( 0.00, -0.36,  0.93), "CP2": ( 0.25, -0.36,  0.90),
    "CP4": ( 0.55, -0.34,  0.77),
    "P1":  (-0.25, -0.72,  0.65), "Pz":  ( 0.00, -0.72,  0.69),
    "P2":  ( 0.25, -0.72,  0.65), "POz": ( 0.00, -0.87,  0.50),
}

TARGET_SRATE = 200
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(TARGET_SRATE * WINDOW_SEC)  # 400


class BCIDataset(Dataset):
    """PyTorch Dataset for BCI IV 2a motor imagery trials.

    Each sample is a 2-second window of 22-channel EEG with a 4-class label.

    Args:
        windows: [N, C, T] array of EEG windows
        labels: [N] array of integer labels (0-3)
        coords: [C, 3] electrode coordinates
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray, coords: np.ndarray):
        self.eeg = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()
        self.coords = torch.from_numpy(coords).float()

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return {
            "eeg": self.eeg[idx],           # [C, T]
            "coords": self.coords,           # [C, 3]
            "label": self.labels[idx],       # scalar
            "num_channels": self.eeg.shape[1],
        }


def _extract_trials(raw, tmin=0.5, tmax=2.5):
    """Extract and preprocess motor imagery trials from MNE Raw.

    Args:
        raw: MNE Raw object with annotations (left_hand, right_hand, feet, tongue)
        tmin: start time relative to cue onset (seconds)
        tmax: end time relative to cue onset (seconds)

    Returns:
        windows: [N, C, T] array
        labels: [N] array (0-indexed: 0=left, 1=right, 2=feet, 3=tongue)
    """
    import mne

    # Pick only EEG channels (drop EOG)
    raw = raw.copy().pick(BCI_CHANNELS)

    # Resample to 200Hz
    if raw.info["sfreq"] != TARGET_SRATE:
        raw.resample(TARGET_SRATE)

    # Common average reference
    raw.set_eeg_reference("average", projection=False)

    # Bandpass filter
    raw.filter(0.5, 75.0, fir_design="firwin", verbose=False)

    # Extract events from annotations
    events, ann_id_map = mne.events_from_annotations(raw, verbose=False)

    # Map annotation names to 0-indexed labels
    name_to_label = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
    # ann_id_map: {'feet': 1, 'left_hand': 2, ...} (annotation name → event code)
    code_to_label = {}
    for ann_name, ann_code in ann_id_map.items():
        clean_name = str(ann_name)
        if clean_name in name_to_label:
            code_to_label[ann_code] = name_to_label[clean_name]

    # Filter events to only MI classes
    mi_event_ids = list(code_to_label.keys())

    epochs = mne.Epochs(
        raw, events, event_id=mi_event_ids,
        tmin=tmin, tmax=tmax - 1.0 / TARGET_SRATE,
        baseline=None, preload=True, verbose=False,
    )

    data = epochs.get_data()  # [N, C, T]
    event_codes = epochs.events[:, 2]
    labels = np.array([code_to_label[c] for c in event_codes])

    # Pad or trim to exact WINDOW_SAMPLES
    T = data.shape[2]
    if T < WINDOW_SAMPLES:
        pad = np.zeros((data.shape[0], data.shape[1], WINDOW_SAMPLES - T))
        data = np.concatenate([data, pad], axis=2)
    elif T > WINDOW_SAMPLES:
        data = data[:, :, :WINDOW_SAMPLES]

    return data.astype(np.float32), labels


def get_coords() -> np.ndarray:
    """Get 3D electrode coordinates for BCI IV 2a channels.

    Returns:
        [22, 3] numpy array
    """
    return np.array([BCI_COORDS[ch] for ch in BCI_CHANNELS], dtype=np.float32)


def load_bci_subject(
    subject_id: int,
) -> tuple[BCIDataset, BCIDataset, np.ndarray]:
    """Load BCI IV 2a data for a single subject.

    Standard protocol: session 1 = train, session 2 = test.

    Args:
        subject_id: 1-9

    Returns:
        train_dataset, test_dataset, coords [22, 3]
    """
    from moabb.datasets import BNCI2014_001

    ds = BNCI2014_001()
    data = ds.get_data(subjects=[subject_id])
    subject_data = data[subject_id]
    coords = get_coords()

    event_id = ds.event_id  # {'left_hand': 1, 'right_hand': 2, 'feet': 3, 'tongue': 4}

    train_windows_list = []
    train_labels_list = []
    test_windows_list = []
    test_labels_list = []

    for session_name, session in subject_data.items():
        is_train = "0train" in session_name or "session_T" in session_name
        for run_name, raw in session.items():
            windows, labels = _extract_trials(raw)
            if is_train:
                train_windows_list.append(windows)
                train_labels_list.append(labels)
            else:
                test_windows_list.append(windows)
                test_labels_list.append(labels)

    train_windows = np.concatenate(train_windows_list)
    train_labels = np.concatenate(train_labels_list)
    test_windows = np.concatenate(test_windows_list)
    test_labels = np.concatenate(test_labels_list)

    return (
        BCIDataset(train_windows, train_labels, coords),
        BCIDataset(test_windows, test_labels, coords),
        coords,
    )


def load_bci_cross_subject(
    test_subject: int,
) -> tuple[BCIDataset, BCIDataset, np.ndarray]:
    """Load BCI IV 2a for cross-subject (LOSO) evaluation.

    Train on all sessions of 8 subjects, test on all sessions of 1 subject.

    Args:
        test_subject: subject ID to hold out (1-9)

    Returns:
        train_dataset, test_dataset, coords [22, 3]
    """
    from moabb.datasets import BNCI2014_001

    ds = BNCI2014_001()
    all_subjects = ds.subject_list
    coords = get_coords()
    event_id = ds.event_id

    train_w, train_l = [], []
    test_w, test_l = [], []

    for subj in all_subjects:
        data = ds.get_data(subjects=[subj])
        for session_name, session in data[subj].items():
            for run_name, raw in session.items():
                windows, labels = _extract_trials(raw)
                if subj == test_subject:
                    test_w.append(windows)
                    test_l.append(labels)
                else:
                    train_w.append(windows)
                    train_l.append(labels)

    return (
        BCIDataset(np.concatenate(train_w), np.concatenate(train_l), coords),
        BCIDataset(np.concatenate(test_w), np.concatenate(test_l), coords),
        coords,
    )
