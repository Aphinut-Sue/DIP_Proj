import numpy as np
import cv2
import os

from skimage.metrics import structural_similarity as ssim


# -----------------------------
# MSE
# -----------------------------
def compute_mse(img1, img2):

    err = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

    return err


# -----------------------------
# PSNR
# -----------------------------
def compute_psnr(img1, img2):

    mse = compute_mse(img1, img2)

    if mse == 0:
        return float("inf")

    PIXEL_MAX = 255.0

    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    return psnr


# -----------------------------
# SSIM
# -----------------------------
def compute_ssim(img1, img2):

    value = ssim(
        img1,
        img2,
        channel_axis=2,
        data_range=255
    )

    return value


# -----------------------------
# DeltaE (CIE76)
# -----------------------------
def compute_deltaE(img1, img2):

    lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)

    diff = lab1.astype(np.float32) - lab2.astype(np.float32)

    deltaE = np.sqrt(np.sum(diff ** 2, axis=2))

    return np.mean(deltaE)


# -----------------------------
# EVALUATE ALL
# -----------------------------
def evaluate(img1, img2):

    mse_val = compute_mse(img1, img2)
    psnr_val = compute_psnr(img1, img2)
    ssim_val = compute_ssim(img1, img2)
    deltaE_val = compute_deltaE(img1, img2)

    return mse_val, psnr_val, ssim_val, deltaE_val


# -----------------------------
# SAVE REPORT
# -----------------------------
def save_report(path, image_name, shape, mse, psnr, ssim_val, deltaE):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    h, w = shape[:2]

    with open(path, "w") as f:

        f.write("Image Quantization Performance Report\n\n")

        f.write(f"Image: {image_name}\n")
        f.write(f"Resolution: {h} x {w}\n\n")

        f.write(f"MSE    : {mse:.4f}\n")
        f.write(f"PSNR   : {psnr:.4f} dB\n")
        f.write(f"SSIM   : {ssim_val:.4f}\n")
        f.write(f"DeltaE : {deltaE:.4f}\n")

    print(f"Performance report saved → {path}")