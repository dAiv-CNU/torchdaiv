import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image as _to_pil_image


def to_pil_image(tensor, mode=None):
    """Convert a tensor to a PIL image.

    Args:
        tensor (Tensor): Tensor to convert.
        mode (str, optional): The mode for the output image. If None, the mode is inferred from the tensor.

    Returns:
        PIL Image: Converted PIL image.
    """
    if isinstance(tensor, torch.Tensor):
        if tensor.ndim == 4:
            tensor = tensor[0]
        return _to_pil_image(tensor.cpu().clamp(0, 1), mode=mode)
    return tensor


def show_result(
    original_img, encoded_img, decoded_img=None, decoded_img2=None,
    labels=("Original", "Encoded", "Decoded", "Decoded"),
    greyscales=(False, False, False, False),
    figsize=(15, 5)
):
    num_imgs = 2 + (int(decoded_img is not None)) + (int(decoded_img2 is not None))
    fig, axes = plt.subplots(1, num_imgs, figsize=figsize)
    
    if isinstance(original_img, torch.Tensor):
        if original_img.ndim == 4:
            original_img = original_img[0]
        original_img = to_pil_image(original_img.cpu().clamp(0, 1))
    if isinstance(encoded_img, torch.Tensor):
        if encoded_img.ndim == 4:
            encoded_img = encoded_img[0]
        encoded_img = to_pil_image(encoded_img.cpu().clamp(0, 1))
    if decoded_img is not None and isinstance(decoded_img, torch.Tensor):
        if decoded_img.ndim == 4:
            decoded_img = decoded_img[0]
        decoded_img = to_pil_image(decoded_img.cpu().clamp(0, 1))
    if decoded_img2 is not None and isinstance(decoded_img2, torch.Tensor):
        if decoded_img2.ndim == 4:
            decoded_img2 = decoded_img2[0]
        decoded_img2 = to_pil_image(decoded_img2.cpu().clamp(0, 1))

    axes[0].imshow(original_img, cmap="gray" if greyscales[0] else None)
    axes[0].set_title(f"{labels[0]} ({original_img.width}×{original_img.height})")
    axes[0].axis("off")

    axes[1].imshow(encoded_img, cmap="gray" if greyscales[1] else None)
    axes[1].set_title(f"{labels[1]} ({encoded_img.width}×{encoded_img.height})")
    axes[1].axis("off")

    if decoded_img is not None:
        axes[2].imshow(decoded_img, cmap="gray" if greyscales[2] else None)
        axes[2].set_title(f"{labels[2]} ({decoded_img.width}×{decoded_img.height})")
        axes[2].axis("off")

    if decoded_img2 is not None:
        axes[3].imshow(decoded_img2, cmap="gray" if greyscales[3] else None)
        axes[3].set_title(f"{labels[3]} ({decoded_img2.width}×{decoded_img2.height})")
        axes[3].axis("off")

    plt.show()
