import argparse
import os
import scipy.io as sio
from dataset import _standardize_hwc
import numpy as np


def find_first_cube(matdict):
    for k, v in matdict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return v, k
    return None, None


def load_hr(path):
    md = sio.loadmat(path)
    for key in ['msi', 'hsi', 'HSI', 'MSI']:
        if key in md:
            arr = md[key]
            return _standardize_hwc(arr, preferred_channels=None, channel_range=(4, 256)), key
    arr, k = find_first_cube(md)
    if arr is None:
        raise RuntimeError(f"No 3D array found in {path}")
    return _standardize_hwc(arr, preferred_channels=None, channel_range=(4, 256)), k


def load_lr(path):
    md = sio.loadmat(path)
    for key in ['LR', 'lr']:
        if key in md:
            arr = md[key]
            return _standardize_hwc(arr, preferred_channels=None, channel_range=(4, 256)), key
    # fallback try HR keys
    for key in ['msi', 'hsi', 'HSI']:
        if key in md:
            arr = md[key]
            return _standardize_hwc(arr, preferred_channels=None, channel_range=(4, 256)), key + ' (fallback)'
    arr, k = find_first_cube(md)
    if arr is None:
        raise RuntimeError(f"No LR-like 3D array found in {path}")
    return _standardize_hwc(arr, preferred_channels=None, channel_range=(4, 256)), k + ' (fallback-any)'


def scan(split_dir, expected_channels):
    issues = []
    if not os.path.isdir(split_dir):
        print(f"[Skip] {split_dir} not found")
        return issues
    files = [f for f in os.listdir(split_dir) if f.endswith('.mat')]
    for f in sorted(files):
        path = os.path.join(split_dir, f)
        try:
            hr, hr_key = load_hr(path)
            lr, lr_key = load_lr(path)
        except Exception as e:
            issues.append((f, 'load_error', str(e)))
            continue
        hr_c = hr.shape[2]
        lr_c = lr.shape[2]
        prob = []
        if hr_c != expected_channels:
            prob.append(f"HR {hr_c}")
        if lr_c != expected_channels:
            prob.append(f"LR {lr_c}")
        if prob:
            issues.append((f, ','.join(prob), f"hr_key={hr_key}, lr_key={lr_key}"))
    return issues


def main():
    ap = argparse.ArgumentParser(description='Validate dataset channel counts')
    ap.add_argument('--root', required=True, help='Root dataset directory containing train/ and test/')
    ap.add_argument('--expected', type=int, default=31, help='Expected spectral channels')
    ap.add_argument('--train_sub', type=str, default='train', help='Train subdir name')
    ap.add_argument('--test_sub', type=str, default='test', help='Test subdir name')
    args = ap.parse_args()

    for split in ['train', 'test']:
        split_path = os.path.join(args.root, args.train_sub if split=='train' else args.test_sub)
        print(f"Scanning {split} split: {split_path}")
        issues = scan(split_path, args.expected)
        if not issues:
            print(f"  No issues found (all {args.expected} channels).")
        else:
            print(f"  Found {len(issues)} issue files:")
            for name, problem, meta in issues:
                print(f"    {name}: {problem} | {meta}")
    print("Done.")


if __name__ == '__main__':
    main()
