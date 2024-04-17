import torch, torchvision
import torch.nn as nn
import numpy as np
import os, random, tifffile, cv2, json
from PIL import Image
from matplotlib import pyplot as plt


def angle(imag, real):
    return np.arctan2(imag, real)


def random_crop(arr, target_height=512, target_width=512):
    """
    If arr's dimensions are less than target_height or target_width,
    mirror fill the array. Then, randomly slice a target_height x target_width
    section from the filled array.
    """
    height, width = arr.shape

    # Mirror fill for height
    while height < target_height:
        arr = np.vstack((arr, arr[:target_height - height, :]))
        height, width = arr.shape  # Update dimensions

    # Mirror fill for width
    while width < target_width:
        arr = np.hstack((arr, arr[:, :target_width - width]))
        height, width = arr.shape  # Update dimensions

    # Now arr is guaranteed to be at least 512x512
    # Randomly choose a starting point for the slice
    start_height = np.random.randint(0, height - target_height + 1)
    start_width = np.random.randint(0, width - target_width + 1)

    # Slice the 512x512 section
    sliced_section = arr[start_height:start_height + target_height, start_width:start_width + target_width]

    return sliced_section


def normalize_data(image):
    # Normalize the image data to 0-255
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm_image


def normalization(tensor):
    min_value = tensor.min().item()
    max_value = tensor.max().item()

    # Normalize the tensor to (-1, 1)
    normalized_tensor = 2 * (tensor - min_value) / (max_value - min_value) - 1

    return normalized_tensor


def complex_add(noise, img):
    noise_complex = np.cos(noise) + 1j * np.sin(noise)
    img_complex = np.cos(img) + 1j * np.sin(img)
    dint_complex = noise_complex * img_complex

    return angle(dint_complex.imag, dint_complex.real)


def add_noise(img, noise_root):
    noise_dir_lst = os.listdir(noise_root)
    id = random.randint(0, len(noise_dir_lst) - 1)

    noise_path = os.path.join(noise_root, noise_dir_lst[id], 'noise_collected.tif')
    noise_less_path = os.path.join(noise_root, noise_dir_lst[id], 'noise_collected_masked.tif')

    try:
        noise = tifffile.imread(noise_path)
        noise_less = tifffile.imread(noise_less_path)
    except Exception as e:
        print(f"Failed to read '{noise_path}'. Error: {e}. Sub-folder: {noise_dir_lst[id]}")

    if random.random() <= 0.5:
        noise_tile = random_crop(noise)
    else:
        noise_tile = random_crop(noise_less)

    # dint_tile = normalize_to_pi(noise_tile + img)
    dint_tile = complex_add(noise_tile, img)

    return dint_tile


def transform_augment(img, noise_root):
    totensor = torchvision.transforms.ToTensor()
    lab_tensor = totensor(img)
    dint = add_noise(img, noise_root)

    # dint_norm = normalize_data(dint)
    # dint_png = cv2.applyColorMap(dint_norm, cv2.COLORMAP_JET)
    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # cv2.imshow("image", dint_png)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    dint_tensor = totensor(dint)
    dint_tensor_norm = normalization(dint_tensor)
    return dint_tensor_norm, lab_tensor


def print_gpu_info():
    print(f'torch version: {torch.__version__}')
    print(f'torchvision version: {torchvision.__version__}')

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()

        print(f"Number of available GPUs: {num_gpus}")

        for gpu_id in range(num_gpus):
            gpu = torch.cuda.get_device_properties(gpu_id)
            print(f"GPU {gpu_id + 1}: {gpu.name}")
            print(f"   Total Memory: {gpu.total_memory / 1e9:.2f} GB")
            print(f"   CUDA Capability: {gpu.major}.{gpu.minor}")

    else:
        print("No GPUs available.")


def save_png(img, img_path):
    img_norm = normalize_data(img)
    c_img_norm = cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)
    cv2.imwrite(img_path, c_img_norm)


def save_tif(img, img_path):
    tif = Image.fromarray(img)
    tif.save(img_path)


def save_loss(loss_history, epoch_history, filename="loss_history.json"):
    with open(filename, 'w') as f:
        json.dump({'loss': loss_history, 'epoch': epoch_history}, f)


def load_loss(filename="loss_history.json"):
    try:
        with open(filename, 'r') as f:
            loss_history = json.load(f)
    except FileNotFoundError:
        loss_history = []
    return loss_history


def plot_loss(loss, epoch_lst, save_path, loss_name=None):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_lst, loss, marker='o', linestyle='-', label=loss_name)

    plt.title("Epoch vs. Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{loss_name}.png'))
    plt.close()


def tensor2img(tensor):
    tensor = tensor.detach().cpu()
    tensor = tensor.squeeze()

    n_dim = tensor.dim()
    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    return img_np


def getDifference(pre, lab):
    pre_complex = torch.cos(pre) + 1j * torch.sin(pre)
    lab_complex = torch.cos(lab) - 1j * torch.sin(lab)
    diff = lab_complex * pre_complex

    return torch.arctan2(diff.imag, diff.real)


class PhaseContinuityLoss(nn.Module):
    def __init__(self):
        super(PhaseContinuityLoss, self).__init__()

    def forward(self, phase):
        """
        Compute the phase continuity loss based on wrapped phase differences.

        Args:
        phase (torch.Tensor): Tensor of phase values with shape (b, c, h, w)

        Returns:
        torch.Tensor: Scalar value of the phase continuity loss
        """
        # Calculate differences along x and y axes
        diff_x = phase[:, :, :, :-1] - phase[:, :, :, 1:]
        diff_y = phase[:, :, :-1, :] - phase[:, :, 1:, :]

        # Apply wrapping correction using atan2
        diff_x = torch.atan2(torch.sin(diff_x), torch.cos(diff_x))
        diff_y = torch.atan2(torch.sin(diff_y), torch.cos(diff_y))

        # Calculate the mean squared error of the phase differences
        loss_x = torch.mean(diff_x ** 2)
        loss_y = torch.mean(diff_y ** 2)

        # Combine losses from both directions
        loss = (loss_x + loss_y) / 2
        return loss





