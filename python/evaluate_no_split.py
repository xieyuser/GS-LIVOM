import os
import os.path as osp
import sys

import numpy as np
from PIL import Image
import sys
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from tqdm import tqdm


def load_image(path):
    return np.array(Image.open(path))

def split_image(image):
    mid = image.shape[1] // 2
    return image[:, :mid], image[:, mid:]

def calculate_metrics(img1, img2, loss_fn):
    # Convert numpy arrays to tensors for LPIPS
    img1_tensor = lpips.im2tensor(img1)  # Add batch dimension
    img2_tensor = lpips.im2tensor(img2)  # Add batch dimension

    ssim_value = ssim(img1[:, :, 0], img2[:, :, 0])
    psnr_value = psnr(img1, img2)
    lpips_value = loss_fn(img1_tensor, img2_tensor).item()

    return ssim_value, psnr_value, lpips_value

def main(folder_path):
    loss_fn = lpips.LPIPS(net='alex')
    
    ssim_values = []
    psnr_values = []
    lpips_values = []

    gt_folder = osp.join(folder_path, "gt")
    render_folder = osp.join(folder_path, "renders")

    for filename in tqdm(os.listdir(gt_folder)[:]):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            gt_image_path = os.path.join(gt_folder, filename)
            gt_image = load_image(gt_image_path)
            render_image_path = os.path.join(render_folder, filename)
            render_image = load_image(render_image_path)

            ssim_value, psnr_value, lpips_value = calculate_metrics(render_image, gt_image, loss_fn)
            ssim_values.append(ssim_value)
            psnr_values.append(psnr_value)
            lpips_values.append(lpips_value)

    mean_ssim = np.mean(ssim_values)
    mean_psnr = np.mean(psnr_values)
    mean_lpips = np.mean(lpips_values)

    print(f"Average SSIM: {round(mean_ssim, 3)}")
    print(f"Average PSNR: {round(mean_psnr, 3)}")
    print(f"Average LPIPS: {round(mean_lpips, 3)}")
if __name__ == "__main__":
    folder_path = sys.argv[1]
    main(folder_path)
