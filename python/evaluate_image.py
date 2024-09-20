import os
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

    for filename in tqdm(os.listdir(folder_path)[:]):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            image = load_image(image_path)
            result_image, ground_truth_image = split_image(image)

            ssim_value, psnr_value, lpips_value = calculate_metrics(result_image, ground_truth_image, loss_fn)
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
