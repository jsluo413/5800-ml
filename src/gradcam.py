from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from .dataset import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def _denorm(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return img.astype(np.float32)


def _load(path):
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tfm(Image.open(path).convert("RGB"))


def make_gradcam_grid(model, target_layer, device, image_paths, class_names, out_path,
                      title="Grad-CAM", cols=4, cell_size=3.0):
    model.eval().to(device)
    cam = GradCAM(model=model, target_layers=[target_layer])

    n = len(image_paths)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_size, rows * cell_size))
    axes = np.array(axes).reshape(rows, cols)

    for i, (path, true_label, tag) in enumerate(image_paths):
        x = _load(path).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
        pred_name = class_names[pred_idx]
        confidence = probs[pred_idx]

        grayscale = cam(input_tensor=x, targets=None)[0]
        rgb = _denorm(x[0])
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

        ax = axes[i // cols, i % cols]
        ax.imshow(overlay)
        ax.set_title(f"{tag}\ntrue={true_label} pred={pred_name} ({confidence:.2f})", fontsize=8)
        ax.axis("off")

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
