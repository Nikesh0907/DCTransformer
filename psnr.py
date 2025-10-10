import numpy as np
import math


def _infer_pixel_max(arr: np.ndarray) -> float:
    # Infer dynamic range; support [0,1] or [0,255]. If already >1, assume 255.
    mx = float(np.nanmax(arr))
    if mx <= 1.0:
        return 1.0
    return 255.0


def _safe_psnr(mse: float, pixel_max: float) -> float:
    if mse <= 0:
        # Perfect reconstruction or numerical underflow
        return 100.0
    return 20.0 * math.log10(pixel_max / math.sqrt(mse))


def MPSNR(img1, img2):
    """Mean PSNR across channels.

    - Accepts inputs in range [0,1] or [0,255]. Mixed ranges are handled by inferring pixel_max from reference img2.
    - Clamps inputs to valid range to avoid negative/overflow artifacts.
    - Computes band-wise PSNR then averages, without early stopping when one band is perfect.
    """
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch in PSNR: {img1.shape} vs {img2.shape}")

    # Infer dynamic range from reference (target) image
    pixel_max = _infer_pixel_max(img2)
    img1 = np.clip(img1, 0.0, pixel_max)
    img2 = np.clip(img2, 0.0, pixel_max)

    if img1.ndim == 2:
        mse = float(np.mean((img1 - img2) ** 2))
        return _safe_psnr(mse, pixel_max)
    elif img1.ndim == 3:
        ch = img1.shape[2]
        psnrs = []
        for i in range(ch):
            mse = float(np.mean((img1[:, :, i] - img2[:, :, i]) ** 2))
            psnrs.append(_safe_psnr(mse, pixel_max))
        return float(np.mean(psnrs))
    else:
        raise ValueError(f"Unsupported input dims for PSNR: ndim={img1.ndim}")



