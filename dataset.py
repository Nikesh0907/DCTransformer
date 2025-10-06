# -*- coding: UTF-8 -*-
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
import scipy.io as sio
import random
import torch
import torch.nn.functional as F
# from kornia.filters import gaussian_blur2d
# from kornia.filters import get_gaussian_kernel2d
# import kornia

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".mat"])


def _standardize_hwc(arr, preferred_channels=None, channel_range=(1, 256)):
    """Return array in (H, W, C) layout.

    Heuristics:
      1. If already (H,W,C) with C within channel_range and H,W >= C -> keep.
      2. Try all permutations; pick one where last axis fits preferred_channels (if given) else channel_range,
         and the first two dims are >= last dim.
      3. Fallback: treat smallest axis as channels, reorder others preserving relative order.
    """
    import itertools
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    def is_ok(shape):
        h,w,c = shape
        return channel_range[0] <= c <= channel_range[1] and h >= c and w >= c

    # Case 1
    if is_ok(arr.shape) and (preferred_channels is None or arr.shape[2] == preferred_channels):
        return arr

    # Case 2: search permutations
    best = None
    for perm in itertools.permutations(range(3)):
        cand = arr.transpose(perm)
        h,w,c = cand.shape
        if preferred_channels is not None and c != preferred_channels:
            continue
        if is_ok((h,w,c)):
            best = cand
            break
    if best is not None:
        return best

    # Case 3: fallback smallest axis as channels
    axes = list(range(3))
    sizes = list(arr.shape)
    ch_axis = int(np.argmin(sizes))
    other_axes = [a for a in axes if a != ch_axis]
    cand = arr.transpose(other_axes[0], other_axes[1], ch_axis)
    return cand


def load_img(filepath):
    """Load high-resolution hyperspectral cube.

    Accepts several possible keys to make raw dataset ingestion easier.
    Expected priority: 'msi' (repo convention) then 'hsi' then first 3D array.
    """
    xmat = sio.loadmat(filepath)
    for k in ['msi', 'hsi', 'HSI', 'MSI']:
        if k in xmat:
            cube = xmat[k]
            break
    else:
        # fallback: pick first ndarray with 3 dims
        cube = None
        for v in xmat.values():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                cube = v
                break
        if cube is None:
            raise KeyError(f"No hyperspectral key found in {filepath}")
    # Standardize to (H,W,C)
    cube = _standardize_hwc(cube, preferred_channels=None, channel_range=(4,256))
    cube = torch.tensor(cube).float()
    return cube

def load_img1(filepath):
    """Load RGB guidance image.

    Uses existing RGB if present; otherwise derives from hyperspectral cube via load_img to ensure orientation.
    """
    xmat = sio.loadmat(filepath)
    if 'RGB' in xmat:
        rgb = xmat['RGB']
    elif 'rgb' in xmat:
        rgb = xmat['rgb']
    else:
        cube = load_img(filepath)  # torch (H,W,C)
        c = cube.shape[2]
        indices = [int(c*0.15), int(c*0.5), int(c*0.85)]
        rgb = cube[:, :, indices].numpy()
    rgb = _standardize_hwc(rgb, preferred_channels=3, channel_range=(3,4))
    rgb = torch.tensor(rgb).float()
    return rgb

def load_img2(filepath):
    xmat = sio.loadmat(filepath)
    if 'blur' in xmat:
        arr = xmat['blur']
    else:
        # fallback to hyperspectral cube if blur not present (will be sub-sampled later)
        for k in ['msi', 'hsi', 'HSI']:
            if k in xmat:
                arr = xmat[k]
                break
        else:
            raise KeyError(f"No 'blur' key in {filepath}")
    arr = _standardize_hwc(arr, preferred_channels=None, channel_range=(4,256))
    return torch.tensor(arr).float()

def load_img3(filepath):
    xmat = sio.loadmat(filepath)
    if 'LR' in xmat:
        lr = xmat['LR']
    else:
        # If low-res not provided, attempt naive downscale to half (for robustness in quick experiments)
        for k in ['msi', 'hsi', 'HSI']:
            if k in xmat:
                hs = xmat[k]
                # simple 2x downscale using stride slicing
                lr = hs[::2, ::2, :]
                break
        else:
            raise KeyError(f"No 'LR' key in {filepath}")
    # Robust reordering: prefer an axis of size close to 31 as spectral
    arr = np.asarray(lr)
    if arr.ndim != 3:
        raise ValueError(f"LR array is not 3D in {filepath}, shape={arr.shape}")
    target = 31
    axes = list(range(3))
    sizes = arr.shape
    # If already channel-last and size plausible, keep
    if sizes[2] in (target, 28, 29, 30, 32, 24, 12):
        pass
    else:
        # Find axis closest to target (within reasonable bound)
        diffs = [abs(s - target) for s in sizes]
        spec_axis = int(np.argmin(diffs))
        if spec_axis != 2:
            order = [a for a in axes if a != spec_axis] + [spec_axis]
            arr = arr.transpose(order)
    arr = _standardize_hwc(arr, preferred_channels=None, channel_range=(4,256))
    return torch.tensor(arr).float()


# def my_gaussian_blur2d(input, kernel_size, sigma, border_type = 'reflect'):
#
#     kernel = torch.unsqueeze(get_gaussian_kernel2d(kernel_size, sigma, force_even=True), dim=0)
#     # print(kernel)
#
#     return kornia.filters.filter2d(input, kernel, border_type)

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir1, image_dir2, image_dir3, upscale_factor, patch_size, input_transform=None, train_prop: float = 1.0, virtual_length: int = 20000):
        super(DatasetFromFolder, self).__init__()

        self.patch_size = patch_size
        self.image_filenames1 = [join(image_dir1, x) for x in listdir(image_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(image_dir2, x) for x in listdir(image_dir2) if is_image_file(x)]
        self.image_filenames3 = [join(image_dir3, x) for x in listdir(image_dir3) if is_image_file(x)]
        self.image_filenames1 = sorted(self.image_filenames1)
        self.image_filenames2 = sorted(self.image_filenames2)
        self.image_filenames3 = sorted(self.image_filenames3)

        # Clamp proportion
        if train_prop <= 0 or train_prop > 1:
            raise ValueError(f"train_prop must be in (0,1], got {train_prop}")
        total_imgs = min(len(self.image_filenames1), len(self.image_filenames2), len(self.image_filenames3))
        use_imgs = max(1, int(total_imgs * train_prop))
        self.image_filenames1 = self.image_filenames1[:use_imgs]
        self.image_filenames2 = self.image_filenames2[:use_imgs]
        self.image_filenames3 = self.image_filenames3[:use_imgs]

        self.base_count = len(self.image_filenames1)
        # Virtual number of samples (random patches) per epoch
        if virtual_length <= 0:
            raise ValueError("virtual_length must be > 0")
        self.lens = virtual_length

        self.xs = [load_img(p) for p in self.image_filenames1]
        self.ys = [load_img1(p) for p in self.image_filenames2]
        self.x_blurs = [load_img2(p) for p in self.image_filenames3]

        self.upscale_factor = upscale_factor
        self.input_transform = input_transform

    def __getitem__(self, index):
        ind = index % self.base_count
        img = self.xs[ind]
        img2 = self.ys[ind]
        img3 = self.x_blurs[ind]
        upscale_factor = self.upscale_factor
        H, W = img.shape[0], img.shape[1]
        if self.patch_size > H or self.patch_size > W:
            raise ValueError(f"Patch size {self.patch_size} exceeds image dimensions {(H, W)}. Reduce patch_size or provide larger images.")
        w = np.random.randint(0, H - self.patch_size + 1)
        h = np.random.randint(0, W - self.patch_size + 1)
        X = img[w:w+self.patch_size, h:h+self.patch_size, :]
        Y = img2[w:w+self.patch_size, h:h+self.patch_size, :]

        # Z = my_gaussian_blur2d(X.unsqueeze(0), (8, 8), (2, 2)).squeeze(0)
        Z = img3[int(w+upscale_factor/2):w+self.patch_size:upscale_factor, int(h+upscale_factor/2):h+self.patch_size:upscale_factor, :]

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        # Random rotation
        X = torch.rot90(X, rotTimes, [0,1])
        Y = torch.rot90(Y, rotTimes, [0,1])
        Z = torch.rot90(Z, rotTimes, [0,1])

        # Random vertical Flip
        for j in range(vFlip):
            X = X.flip(1)
            Y = Y.flip(1)
            Z = Z.flip(1)

        # Random Horizontal Flip
        for j in range(hFlip):
            X = X.flip(0)
            Y = Y.flip(0)
            Z = Z.flip(0)

        X = X.permute(2,0,1)
        Y = Y.permute(2, 0, 1)
        Z = Z.permute(2, 0, 1)

        return Z, Y, X

    def __len__(self):
        return self.lens


class DatasetFromFolder2(data.Dataset):
    def __init__(self, image_dir1, image_dir2, image_dir3, input_transform=None):
        super(DatasetFromFolder2, self).__init__()
        self.image_filenames1 = [join(image_dir1, x) for x in listdir(image_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(image_dir2, x) for x in listdir(image_dir2) if is_image_file(x)]
        self.image_filenames3 = [join(image_dir3, x) for x in listdir(image_dir3) if is_image_file(x)]
        self.image_filenames1 = sorted(self.image_filenames1)
        self.image_filenames2 = sorted(self.image_filenames2)
        self.image_filenames3 = sorted(self.image_filenames3)
        # self.upscale_factor = upscale_factor
        self.input_transform = input_transform

        self.xs = []
        self.xs_name = []
        for img in self.image_filenames1:
            self.xs.append(load_img(img))
            self.xs_name.append(img)

        self.ys = []
        for img in self.image_filenames2:
            self.ys.append(load_img1(img))

        self.zs = []
        for img in self.image_filenames3:
            self.zs.append(load_img3(img))

    def __getitem__(self, index):
        X = self.xs[index]
        Y = self.ys[index]
        Z = self.zs[index]

        # upscale_factor = self.upscale_factor

        # Z = F.interpolate(X.permute(2, 0, 1).unsqueeze(0), scale_factor=1.0 / upscale_factor, mode='bicubic',
        #                     align_corners=False, recompute_scale_factor=False).squeeze(0).permute(1, 2, 0)

        #
        # Z = my_gaussian_blur2d(X.unsqueeze(0), (8, 8), (2, 2)).squeeze(0)
        # Z = Z[int(upscale_factor/2)::upscale_factor, int(upscale_factor/2)::upscale_factor, :]




        X = X.permute(2, 0, 1)
        # Y = Y.permute(2, 0, 1)
        Z = Z.permute(2, 0, 1)
        Y = Y.permute(2, 0, 1)

        return Z, Y, X, self.xs_name[index]


    def __len__(self):
        return len(self.image_filenames1)