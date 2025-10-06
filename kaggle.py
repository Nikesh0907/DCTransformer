import os
import scipy.io as sio
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
import h5py
from typing import Tuple

# CONFIG
raw_dir = Path('/kaggle/input/cave-hsi')
out_root = Path('/kaggle/working/CAVEdata12')
upscale_factor = 8
train_count = 20   # adjust
force_regen = True  # if True, will overwrite existing processed files
np.random.seed(42)

# Gather files
files = sorted([f for f in raw_dir.glob('*.mat')])
assert len(files) >= train_count + 1, "Not enough files for chosen split."

train_files = files[:train_count]
test_files = files[train_count:]

# Make dirs
for p in [
    out_root/'train'/'X',
    out_root/'train'/'Y',
    out_root/'train'/'X_blur',
    out_root/'test'/'X',
    out_root/'test'/'Y',
    out_root/'test'/'Z'
]:
    p.mkdir(parents=True, exist_ok=True)

def standardize_hwc(arr: np.ndarray) -> np.ndarray:
    """Ensure array is (H,W,C). Accepts (H,W,C) or (C,H,W)."""
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")
    # If it's channel-first (C,H,W) with small C
    if arr.shape[0] <= 64 and arr.shape[1] >= 64 and arr.shape[2] >= 64:
        return np.transpose(arr, (1,2,0))
    # If already channel-last
    return arr

def make_rgb_from_hsi(hsi: np.ndarray) -> np.ndarray:
    hsi = standardize_hwc(hsi)
    c = hsi.shape[2]
    if c < 3:
        raise ValueError(f"Cannot derive RGB from cube with {c} channels")
    idx = [int(c*0.15), int(c*0.5), int(c*0.85)]
    rgb = hsi[:, :, idx].astype(np.float32)
    # normalize each channel independently to [0,1]
    for i in range(3):
        ch = rgb[:,:,i]
        mn, mx = ch.min(), ch.max()
        rgb[:,:,i] = 0 if mx <= mn else (ch - mn) / (mx - mn)
    return rgb

def to_uint16_if_large(x):
    # Optional: keep as float (model expects float). We'll keep float.
    return x

def load_hsi_cube(path):
    """Robust loader for HSI cube supporting v7.3 (HDF5) and older MAT versions.

    Returns np.ndarray (H, W, C) in float32.
    Accepts keys in priority order: 'hsi', 'msi', 'HSI', 'MSI'.
    """
    keys_priority = ['hsi', 'msi', 'HSI', 'MSI']
    try:
        mat = sio.loadmat(path)
        for k in keys_priority:
            if k in mat:
                arr = mat[k]
                break
        else:
            raise KeyError(f"No expected hyperspectral key in {path}")
    except NotImplementedError:
        # v7.3 HDF5 case
        with h5py.File(path, 'r') as f:
            found = None
            for k in keys_priority:
                if k in f:
                    found = k
                    break
            if found is None:
                # fallback: pick first 3D dataset
                for k in f.keys():
                    ds = f[k]
                    if isinstance(ds, h5py.Dataset) and ds.ndim == 3:
                        found = k
                        break
            if found is None:
                raise KeyError(f"No hyperspectral 3D dataset in {path}")
            arr = np.array(f[found])
            # Heuristic: ensure last dim is spectral (C). If first dim looks like channels (<= 64) and last > 128, transpose.
            if arr.ndim == 3 and arr.shape[0] < 64 and arr.shape[0] <=  arr.shape[2]:
                # assume MATLAB saved (C, H, W)
                if arr.shape[1] == arr.shape[2]:  # square spatial dims ambiguous; skip
                    pass
                else:
                    arr = np.transpose(arr, (1, 2, 0))
            # Else assume already (H,W,C)
    # Standardize orientation now
    arr = standardize_hwc(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def save_train_sample(src_path, dst_stem):
    x_path = out_root/'train'/'X'/dst_stem
    y_path = out_root/'train'/'Y'/dst_stem
    b_path = out_root/'train'/'X_blur'/dst_stem
    if not force_regen and x_path.exists() and y_path.exists() and b_path.exists():
        return
    hsi = load_hsi_cube(src_path)  # (H,W,C)
    rgb = make_rgb_from_hsi(hsi)   # (H,W,3)
    blur = gaussian_filter(hsi, sigma=(2,2,0))  # same orientation
    sio.savemat(x_path, {'msi': hsi})
    sio.savemat(y_path, {'RGB': rgb})
    sio.savemat(b_path, {'blur': blur})

def save_test_sample(src_path, dst_stem):
    x_path = out_root/'test'/'X'/dst_stem
    y_path = out_root/'test'/'Y'/dst_stem
    z_path = out_root/'test'/'Z'/dst_stem
    if not force_regen and x_path.exists() and y_path.exists() and z_path.exists():
        return
    hsi = load_hsi_cube(src_path)
    rgb = make_rgb_from_hsi(hsi)
    s = upscale_factor
    H, W, _ = hsi.shape
    lr = hsi[int(s/2)::s, int(s/2)::s, :]
    sio.savemat(x_path, {'msi': hsi})
    sio.savemat(y_path, {'RGB': rgb})
    sio.savemat(z_path, {'LR': lr})

# Process train
for f in train_files:
    save_train_sample(f, f.name)

# Process test
for f in test_files:
    save_test_sample(f, f.name)

print("Done. Root prepared at:", out_root)
print("Train HSI count:", len(train_files), "Test HSI count:", len(test_files))
sample_train = (out_root/'train'/'X'/train_files[0].name)
mat_debug = sio.loadmat(sample_train)
print('DEBUG train X sample keys:', mat_debug.keys())
print('DEBUG train X sample msi shape:', mat_debug['msi'].shape)